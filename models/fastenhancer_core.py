from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations


class _StridedConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> None:
        if kernel_size % stride:
            raise ValueError("kernel_size must be divisible by stride")
        self.original_stride = int(stride)
        self.original_padding = int(padding)
        super().__init__(
            in_channels * stride,
            out_channels,
            kernel_size // stride,
            bias=False,
        )

    def forward(self, values: Tensor) -> Tensor:
        values = F.pad(values, (self.original_padding, self.original_padding))
        batch, channels, frequencies = values.shape
        values = values.view(
            batch,
            channels,
            frequencies // self.original_stride,
            self.original_stride,
        )
        values = values.permute(0, 3, 1, 2).reshape(
            batch,
            channels * self.original_stride,
            frequencies // self.original_stride,
        )
        return super().forward(values)


class _ScaledConvTranspose1d(nn.ConvTranspose1d):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, values: Tensor) -> Tensor:
        weights = F.normalize(self.weight, dim=(0, 1, 2)) * self.scale
        return F.conv_transpose1d(
            values,
            weights,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.groups,
            dilation=self.dilation,
        )


@torch.no_grad()
def _materialize_strided_conv(module: _StridedConv1d) -> nn.Conv1d:

    stride = module.original_stride
    packed_channels = module.in_channels
    if packed_channels % stride:
        raise ValueError("packed input channels must be divisible by the unshuffle stride")
    input_channels = packed_channels // stride
    packed_kernel = module.kernel_size[0]
    deployed = nn.Conv1d(
        input_channels,
        module.out_channels,
        packed_kernel * stride,
        stride=stride,
        padding=module.original_padding,
        dilation=module.dilation,
        groups=module.groups,
        bias=module.bias is not None,
        padding_mode=module.padding_mode,
        device=module.weight.device,
        dtype=module.weight.dtype,
    )



    weight = (
        module.weight.reshape(
            module.out_channels,
            stride,
            input_channels,
            packed_kernel,
        )
        .permute(0, 2, 3, 1)
        .reshape(module.out_channels, input_channels, packed_kernel * stride)
    )
    deployed.weight.copy_(weight)
    if module.bias is not None:
        deployed.bias.copy_(module.bias)
    return deployed


@torch.no_grad()
def _materialize_scaled_conv_transpose(
    module: _ScaledConvTranspose1d,
) -> nn.ConvTranspose1d:

    deployed = nn.ConvTranspose1d(
        module.in_channels,
        module.out_channels,
        module.kernel_size,
        stride=module.stride,
        padding=module.padding,
        output_padding=module.output_padding,
        groups=module.groups,
        bias=module.bias is not None,
        dilation=module.dilation,
        padding_mode=module.padding_mode,
        device=module.weight.device,
        dtype=module.weight.dtype,
    )
    deployed.weight.copy_(F.normalize(module.weight, dim=(0, 1, 2)) * module.scale)
    if module.bias is not None:
        deployed.bias.copy_(module.bias)
    return deployed


class _ChannelsLastBatchNorm(nn.BatchNorm1d):
    def forward(self, values: Tensor) -> Tensor:
        frames, batch, frequencies, channels = values.shape
        values = values.view(frames * batch * frequencies, channels, 1)
        return super().forward(values).view(frames, batch, frequencies, channels)


@torch.no_grad()
def _fold_linear_batch_norm(linear: nn.Linear, normalization: nn.BatchNorm1d) -> None:
    scale = normalization.weight / torch.sqrt(normalization.running_var + normalization.eps)
    if linear.out_features % normalization.num_features:
        raise ValueError("linear output must contain whole normalization channel groups")
    repeats = linear.out_features // normalization.num_features
    scale = scale.repeat(repeats)
    running_mean = normalization.running_mean.repeat(repeats)
    normalization_bias = normalization.bias.repeat(repeats)
    bias = (
        linear.bias
        if linear.bias is not None
        else torch.zeros(
            linear.out_features, device=linear.weight.device, dtype=linear.weight.dtype
        )
    )
    linear.weight = nn.Parameter(linear.weight * scale[:, None])
    linear.bias = nn.Parameter((bias - running_mean) * scale + normalization_bias)


def _position(channels: int, frequencies: int) -> Tensor:
    frequency_grid = torch.arange(1, frequencies + 1, dtype=torch.float32) * (math.pi / frequencies)
    channel_grid = torch.linspace(0.0, math.log(frequencies - 1), channels // 2).exp()
    grid = frequency_grid[:, None] * channel_grid[None]
    return torch.cat((grid.sin(), grid.cos()), dim=1)


class _Attention(nn.Module):
    def __init__(self, channels: int, heads: int) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("attention channels must be divisible by heads")
        self.channels = channels // heads
        self.heads = heads
        self.qkv = nn.Linear(channels, 3 * channels, bias=False)

    def forward(self, values: Tensor) -> Tensor:
        batch_time, frequencies, _ = values.shape
        qkv = (
            self.qkv(values)
            .reshape(batch_time, frequencies, self.heads, 3 * self.channels)
            .transpose(1, 2)
        )
        query, key, value = qkv.split(self.channels, dim=-1)
        attended = F.scaled_dot_product_attention(query, key, value)
        return attended.transpose(1, 2).reshape(batch_time, frequencies, self.heads * self.channels)


class _RNNFormerBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        frequencies: int,
        heads: int,
        epsilon: float,
        position: bool,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.frequencies = int(frequencies)
        self.rnn = nn.GRU(channels, channels)
        self.rnn_fc = nn.Linear(channels, channels, bias=False)
        self.rnn_post_norm = _ChannelsLastBatchNorm(channels, epsilon)
        self.attn = _Attention(channels, heads)
        self.attn_fc = nn.Linear(channels, channels, bias=False)
        self.attn_post_norm = _ChannelsLastBatchNorm(channels, epsilon)
        self.pe = nn.Parameter(_position(channels, frequencies)) if position else None
        weight_norm(self.rnn, name="weight_ih_l0")
        weight_norm(self.rnn, name="weight_hh_l0")
        weight_norm(self.attn.qkv)

    def forward(self, values: Tensor, state: Tensor | None) -> tuple[Tensor, Tensor]:
        frames, batch, frequencies, channels = values.shape
        residual = values
        values, state = self.rnn(values.view(frames, frequencies * batch, channels), state)
        values = values.view(frames, batch, frequencies, channels)
        values = self.rnn_post_norm(self.rnn_fc(values)).add_(residual)
        if self.pe is not None:
            values = values.add_(self.pe)
        residual = values
        values = self.attn(values.view(frames * batch, frequencies, channels)).view(
            frames, batch, frequencies, channels
        )
        values = self.attn_post_norm(self.attn_fc(values))
        return values.add_(residual), state


def _filterbank(input_frequencies: int, output_frequencies: int) -> tuple[nn.Linear, nn.Linear]:
    pre = nn.Linear(input_frequencies, output_frequencies, bias=False)
    post = nn.Linear(output_frequencies, input_frequencies, bias=False)
    centres = torch.linspace(0, 8000, output_frequencies)
    delta = 8000 / output_frequencies
    frequencies = torch.linspace(0, 8000, input_frequencies)
    down = (centres[1:, None] - frequencies[None]) / delta
    down = F.pad(down, (0, 0, 0, 1), value=1.0)
    up = (frequencies[None] - centres[:-1, None]) / delta
    up = F.pad(up, (0, 0, 1, 0), value=1.0)
    pre_weight = torch.maximum(up.new_zeros(1), torch.minimum(down, up))
    post_weight = pre_weight.transpose(0, 1)
    pre_weight = pre_weight / pre_weight.sum(1, keepdim=True)
    post_weight = post_weight / post_weight.sum(1, keepdim=True)
    delattr(pre, "weight")
    delattr(post, "weight")
    pre.register_buffer("weight", pre_weight.contiguous().clone())
    post.register_buffer("weight", post_weight.contiguous().clone())
    return pre, post


class FastEnhancerModel(nn.Module):


    def __init__(
        self,
        *,
        channels: int,
        state_channels: int,
        state_frequencies: int,
        state_blocks: int,
        kernels: tuple[int, ...],
        n_fft: int = 512,
        hop_len: int = 256,
        win_len: int = 512,
        sample_rate: int = 16000,
        compression: float = 0.3,
        attention_heads: int = 4,
        epsilon: float = 1.0e-5,
    ) -> None:
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop_len = int(hop_len)
        self.win_len = int(win_len)
        self.sample_rate = int(sample_rate)
        self.compression = float(compression)
        self.epsilon = float(epsilon)
        self.channels = int(channels)
        self.state_channels = int(state_channels)
        self.state_frequencies = int(state_frequencies)
        self.state_blocks = int(state_blocks)
        self.training_weight_norm = True
        stride = 4

        self.enc_pre = nn.Sequential(
            _StridedConv1d(
                2,
                channels,
                kernels[0],
                stride,
                (kernels[0] - stride) // 2,
            ),
            nn.BatchNorm1d(channels),
            nn.SiLU(inplace=True),
        )
        self.encoder = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel,
                        padding=(kernel - 1) // 2,
                        bias=False,
                    ),
                    nn.BatchNorm1d(channels),
                    nn.SiLU(inplace=True),
                )
                for kernel in kernels[1:]
            ]
        )
        pre, post = _filterbank(n_fft // 2 // stride, state_frequencies)
        self.rf_pre = nn.Sequential(
            pre,
            nn.Conv1d(channels, state_channels, 1, bias=False),
            nn.BatchNorm1d(state_channels),
        )
        self.rf_block = nn.ModuleList(
            [
                _RNNFormerBlock(
                    state_channels,
                    state_frequencies,
                    attention_heads,
                    epsilon,
                    position=index == 0,
                )
                for index in range(state_blocks)
            ]
        )
        self.rf_post = nn.Sequential(
            post,
            nn.Conv1d(state_channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.decoder = nn.ModuleList()
        for kernel in reversed(kernels[1:]):
            self.decoder.append(
                nn.Sequential(
                    nn.Conv1d(2 * channels, channels, 1, bias=False),
                    nn.BatchNorm1d(channels),
                    nn.SiLU(inplace=True),
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel,
                        padding=(kernel - 1) // 2,
                        bias=False,
                    ),
                    nn.BatchNorm1d(channels),
                    nn.SiLU(inplace=True),
                )
            )
        upsample = _ScaledConvTranspose1d(
            channels,
            2,
            kernels[0],
            stride=stride,
            padding=(kernels[0] - stride) // 2,
        )
        self.dec_post = nn.Sequential(
            nn.Conv1d(2 * channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.SiLU(inplace=True),
            upsample,
        )

    def _model_forward(
        self, compressed: Tensor, states: tuple[Tensor, ...] = ()
    ) -> tuple[Tensor, tuple[Tensor, ...]]:
        batch, frequencies, frames, _ = compressed.shape
        values = compressed.permute(0, 2, 3, 1).reshape(batch * frames, 2, frequencies)
        values = self.enc_pre(values)
        skips = [values]
        for encoder in self.encoder:
            values = encoder(values)
            skips.append(values)
        values = self.rf_pre(values)
        values = (
            values.view(batch, frames, self.state_channels, self.state_frequencies)
            .permute(1, 0, 3, 2)
            .contiguous()
        )
        state_inputs: tuple[Tensor | None, ...] = states if states else (None,) * len(self.rf_block)
        output_states = []
        for block, state in zip(self.rf_block, state_inputs):
            values, state = block(values, state)
            output_states.append(state)
        values = values.permute(1, 0, 3, 2).reshape(
            batch * frames, self.state_channels, self.state_frequencies
        )
        values = self.rf_post(values)
        for decoder in self.decoder:
            values = decoder(torch.cat((values, skips.pop()), dim=1))
        values = self.dec_post(torch.cat((values, skips.pop()), dim=1))
        mask = values.reshape(batch, frames, 2, frequencies).permute(0, 3, 1, 2).contiguous()
        return mask, tuple(output_states)

    def training_forward(self, noisy: Tensor) -> tuple[Tensor, Tensor]:
        window = torch.hann_window(self.win_len, device=noisy.device, dtype=noisy.dtype)
        spectrum = torch.stft(
            noisy,
            self.n_fft,
            hop_length=self.hop_len,
            win_length=self.win_len,
            window=window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        body = spectrum[:, :-1]
        magnitude = body.abs().clamp_min(self.epsilon)
        compressed = body * magnitude.pow(self.compression - 1.0)
        mask, _ = self._model_forward(torch.view_as_real(compressed))
        enhanced_compressed = compressed * torch.view_as_complex(mask)
        enhanced_magnitude = enhanced_compressed.abs()
        enhanced = enhanced_compressed * enhanced_magnitude.pow(1.0 / self.compression - 1.0)
        enhanced = F.pad(enhanced, (0, 0, 0, 1))
        waveform = torch.istft(
            enhanced,
            self.n_fft,
            hop_length=self.hop_len,
            win_length=self.win_len,
            window=window,
            center=True,
            normalized=False,
            onesided=True,
            length=noisy.shape[-1],
        )
        return waveform, torch.view_as_real(enhanced_compressed)

    def forward(self, waveform: Tensor) -> Tensor:
        return self.training_forward(waveform)[0]

    def initial_states(
        self,
        batch_size: int = 1,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> tuple[Tensor, ...]:
        return tuple(
            torch.zeros(
                1,
                batch_size * self.state_frequencies,
                self.state_channels,
                device=device,
                dtype=dtype,
            )
            for _ in self.rf_block
        )

    initialize_state = initial_states

    def forward_frame(self, spectrum_frame: Tensor, *states: Tensor) -> tuple[Tensor, ...]:
        if spectrum_frame.ndim == 3:
            spectrum_frame = spectrum_frame.unsqueeze(2)
        body = spectrum_frame[:, :-1]
        magnitude = body.square().sum(-1, keepdim=True).clamp_min(self.epsilon**2).sqrt()
        compressed = body * magnitude.pow(self.compression - 1.0)
        mask, output_states = self._model_forward(compressed, tuple(states))
        enhanced = torch.stack(
            (
                compressed[..., 0] * mask[..., 0] - compressed[..., 1] * mask[..., 1],
                compressed[..., 0] * mask[..., 1] + compressed[..., 1] * mask[..., 0],
            ),
            dim=3,
        )
        magnitude = enhanced.square().sum(-1, keepdim=True).clamp_min(self.epsilon**2).sqrt()
        enhanced = enhanced * magnitude.pow(1.0 / self.compression - 1.0)
        return (F.pad(enhanced, (0, 0, 0, 0, 0, 1)), *output_states)

    @torch.no_grad()
    def remove_weight_reparameterizations(self) -> None:

        if self.training_weight_norm:
            for block in self.rf_block:
                remove_parametrizations(block.rnn, "weight_ih_l0")
                remove_parametrizations(block.rnn, "weight_hh_l0")
                remove_parametrizations(block.attn.qkv, "weight")
                _fold_linear_batch_norm(block.rnn_fc, block.rnn_post_norm)
                block.rnn_post_norm = nn.Identity()
                _fold_linear_batch_norm(block.attn_fc, block.attn_post_norm)
                block.attn_post_norm = nn.Identity()
            self.training_weight_norm = False
        if isinstance(self.enc_pre[0], _StridedConv1d):
            self.enc_pre[0] = _materialize_strided_conv(self.enc_pre[0])
        if isinstance(self.dec_post[-1], _ScaledConvTranspose1d):
            self.dec_post[-1] = _materialize_scaled_conv_transpose(self.dec_post[-1])


__all__ = ["FastEnhancerModel"]
