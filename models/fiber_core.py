from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations

from models.fastenhancer_core import (
    _ScaledConvTranspose1d,
    _StridedConv1d,
    _filterbank,
    _materialize_scaled_conv_transpose,
    _materialize_strided_conv,
)


def _linear_filterbank(input_bins: int, output_bins: int) -> Tensor:
    centres = torch.linspace(0, input_bins - 1, output_bins)
    frequencies = torch.arange(input_bins, dtype=torch.float32)
    delta = (input_bins - 1) / (output_bins - 1)
    weights = (1.0 - (frequencies[None] - centres[:, None]).abs() / delta).clamp_min(0.0)
    return weights / weights.sum(1, keepdim=True).clamp_min(1.0e-12)


def _sinusoidal_position(frequencies: int, channels: int) -> Tensor:
    positions = torch.arange(1, frequencies + 1, dtype=torch.float32) * (math.pi / frequencies)
    rates = torch.linspace(0.0, math.log(max(frequencies - 1, 1)), channels // 2).exp()
    phases = positions[:, None] * rates[None]
    embedding = torch.cat((phases.sin(), phases.cos()), dim=1)
    if embedding.shape[1] < channels:
        embedding = F.pad(embedding, (0, channels - embedding.shape[1]))
    return embedding


class _ChannelsLastBatchNorm(nn.Module):
    def __init__(self, channels: int, epsilon: float) -> None:
        super().__init__()
        self.normalization = nn.BatchNorm1d(channels, eps=epsilon)

    def forward(self, values: Tensor) -> Tensor:
        shape = values.shape
        return self.normalization(values.reshape(-1, shape[-1])).reshape(shape)


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


class _FrequencyAttention(nn.Module):
    def __init__(self, channels: int, heads: int) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("state channels must be divisible by attention heads")
        self.channels = int(channels)
        self.heads = int(heads)
        self.head_channels = channels // heads
        self.qkv = nn.Linear(channels, 3 * channels, bias=False)

    def forward(self, values: Tensor) -> Tensor:
        batch_time, frequencies, _ = values.shape
        qkv = self.qkv(values).reshape(batch_time, frequencies, self.heads, 3 * self.head_channels)
        qkv = qkv.transpose(1, 2)
        query, key, value = qkv.split(self.head_channels, dim=-1)
        attended = F.scaled_dot_product_attention(query, key, value)
        return attended.transpose(1, 2).reshape(batch_time, frequencies, self.channels)


class _FiberBlock(nn.Module):


    def __init__(
        self,
        channels: int,
        frequencies: int,
        heads: int,
        epsilon: float,
        *,
        position: bool,
        group_size: int,
        group_offset: int,
    ) -> None:
        super().__init__()
        if frequencies % group_size:
            raise ValueError("group_size must divide state_frequencies")
        self.channels = int(channels)
        self.frequencies = int(frequencies)
        self.temporal_tile_size = int(group_size)
        self.temporal_tile_offset = int(group_offset) % self.temporal_tile_size
        self.temporal_tiles = frequencies // self.temporal_tile_size
        self.temporal_channels = channels * self.temporal_tile_size
        self.temporal_state_batches = self.temporal_tiles
        self.temporal_hidden_channels = self.temporal_channels

        self.temporal = nn.GRU(self.temporal_channels, self.temporal_hidden_channels)
        self.temporal_projection = nn.Linear(
            self.temporal_hidden_channels, self.temporal_channels, bias=False
        )
        self.temporal_norm = _ChannelsLastBatchNorm(channels, epsilon)
        self.spectral = _FrequencyAttention(channels, heads)
        self.spectral_projection = nn.Linear(channels, channels, bias=False)
        self.spectral_norm = _ChannelsLastBatchNorm(channels, epsilon)
        self.position = (
            nn.Parameter(_sinusoidal_position(frequencies, channels)) if position else None
        )

    def forward(self, values: Tensor, state: Tensor | None) -> tuple[Tensor, Tensor]:
        frames, batch, frequencies, channels = values.shape
        temporal = torch.roll(values, -self.temporal_tile_offset, dims=2).reshape(
            frames,
            batch * self.temporal_tiles,
            self.temporal_channels,
        )
        temporal, state = self.temporal(temporal, state)
        temporal = self.temporal_projection(temporal).reshape(frames, batch, frequencies, channels)
        temporal = torch.roll(temporal, self.temporal_tile_offset, dims=2)
        values = values + self.temporal_norm(temporal)
        if self.position is not None:
            values = values + self.position

        spectral = self.spectral(values.reshape(frames * batch, frequencies, channels)).reshape(
            frames, batch, frequencies, channels
        )
        values = values + self.spectral_norm(self.spectral_projection(spectral))
        return values, state


def _alias_weight(target: nn.GRU, source: nn.GRU, name: str) -> None:
    target_parameters = target.parametrizations[name]
    source_parameters = source.parametrizations[name]
    target_parameters.original0 = source_parameters.original0
    target_parameters.original1 = source_parameters.original1


class FiberModel(nn.Module):


    def __init__(
        self,
        *,
        channels: int,
        state_channels: int,
        state_frequencies: int,
        state_blocks: int = 3,
        recurrent_weight_banks: int,
        recurrent_weight_bank_assignment: Sequence[int] | None = None,
        n_fft: int = 512,
        hop_len: int = 256,
        win_len: int = 512,
        sample_rate: int = 16000,
        attention_heads: int = 4,
        kernels: Sequence[int] = (8, 3, 3),
        stride: int = 4,
        compression: float = 0.3,
        group_size: int = 4,
        alternate_groups: bool = True,
        gram_observation: bool = True,
        simple_observation: bool = False,
        fastenhancer_filterbank: bool = False,
        fastenhancer_codec: bool = False,
        fastenhancer_state_to_codec_layout: bool = False,
        training_weight_norm: bool = True,
        posterior_readout_dropout: float = 0.0,
        epsilon: float = 1.0e-5,
    ) -> None:
        super().__init__()
        if state_blocks < 1:
            raise ValueError("state_blocks must be positive")
        if not 1 <= recurrent_weight_banks <= state_blocks:
            raise ValueError("recurrent_weight_banks must lie in [1, state_blocks]")
        if not 0.0 <= posterior_readout_dropout < 1.0:
            raise ValueError("posterior_readout_dropout must lie in [0, 1)")
        if gram_observation and simple_observation:
            raise ValueError("Gram and simple observations are mutually exclusive")
        if fastenhancer_codec and (gram_observation or simple_observation):
            raise ValueError("FastEnhancer codec matching requires the two-channel observation")
        if fastenhancer_state_to_codec_layout and not fastenhancer_codec:
            raise ValueError("FastEnhancer state-to-codec layout requires its codec boundary")

        self.n_fft = int(n_fft)
        self.hop_len = int(hop_len)
        self.win_len = int(win_len)
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.state_channels = int(state_channels)
        self.state_readout_channels = int(state_channels)
        self.state_frequencies = int(state_frequencies)
        self.state_blocks = int(state_blocks)
        self.state_topology = "factorized"
        self.attention_heads = int(attention_heads)
        self.stride = int(stride)
        self.compression = float(compression)
        self.gram_observation = bool(gram_observation)
        self.simple_observation = bool(simple_observation)
        self.fastenhancer_filterbank = bool(fastenhancer_filterbank)
        self.fastenhancer_codec = bool(fastenhancer_codec)
        self.fastenhancer_state_to_codec_layout = bool(
            fastenhancer_state_to_codec_layout
        )
        self.temporal_tile_size = int(group_size)
        self.temporal_tile_sizes = (int(group_size),) * state_blocks
        self.temporal_tile_offsets = tuple(
            group_size // 2 if alternate_groups and index % 2 else 0
            for index in range(state_blocks)
        )
        self.temporal_hidden_channels = (state_channels * group_size,) * state_blocks
        self.training_weight_norm = bool(training_weight_norm)
        self.identity_output_initialization = not self.fastenhancer_codec
        self.posterior_readout_dropout = float(posterior_readout_dropout)
        self.output_geometry = "compressed_complex"
        self.epsilon = float(epsilon)
        self.mask_channels = 2
        self.recurrent_weight_banks = int(recurrent_weight_banks)
        if recurrent_weight_bank_assignment is None:
            assignment = tuple(index % recurrent_weight_banks for index in range(state_blocks))
        else:
            assignment = tuple(int(value) for value in recurrent_weight_bank_assignment)
            if len(assignment) != state_blocks:
                raise ValueError("recurrent_weight_bank_assignment must have state_blocks entries")
            if set(assignment) != set(range(recurrent_weight_banks)):
                raise ValueError("recurrent_weight_bank_assignment must use each bank in [0, K)")
        self.recurrent_weight_bank_assignment = assignment
        self.recurrent_weight_bank_sources = tuple(
            assignment.index(bank) for bank in range(recurrent_weight_banks)
        )
        self.recurrent_weight_bank_scope = "gru-weight-ih-and-weight-hh-only"
        self.posterior_dropout = nn.Dropout(posterior_readout_dropout)
        self.decoder_dropout = nn.Dropout(0.0)


        input_projection: nn.Module
        if self.fastenhancer_state_to_codec_layout:
            input_projection = _StridedConv1d(
                2,
                channels,
                kernels[0],
                stride,
                (kernels[0] - stride) // 2,
            )
        else:
            input_projection = nn.Conv1d(
                2,
                channels,
                kernels[0],
                stride=stride,
                padding=(kernels[0] - stride) // 2,
                bias=False,
            )
        self.input = nn.Sequential(
            input_projection,
            nn.BatchNorm1d(channels, eps=epsilon),
            nn.SiLU(inplace=True),
        )
        self.encoder = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel,
                        padding=kernel // 2,
                        bias=False,
                    ),
                    nn.BatchNorm1d(channels, eps=epsilon),
                    nn.SiLU(inplace=True),
                )
                for kernel in kernels[1:]
            ]
        )
        encoded_frequencies = n_fft // 2 // stride
        if self.fastenhancer_filterbank:
            analysis_layer, synthesis_layer = _filterbank(
                encoded_frequencies, state_frequencies
            )
            analysis = analysis_layer.weight.contiguous().clone()
            synthesis = synthesis_layer.weight.contiguous().clone()
        else:
            analysis = _linear_filterbank(encoded_frequencies, state_frequencies)
            synthesis = analysis.transpose(0, 1)
            synthesis = synthesis / synthesis.sum(1, keepdim=True).clamp_min(1.0e-12)
        self.register_buffer("state_analysis", analysis, persistent=True)
        self.register_buffer("state_synthesis", synthesis, persistent=True)
        self.state_in = nn.Sequential(
            nn.Conv1d(channels, state_channels, 1, bias=False),
            nn.BatchNorm1d(state_channels, eps=epsilon),
        )
        self.blocks = nn.ModuleList(
            [
                _FiberBlock(
                    state_channels,
                    state_frequencies,
                    attention_heads,
                    epsilon,
                    position=index == 0,
                    group_size=group_size,
                    group_offset=self.temporal_tile_offsets[index],
                )
                for index in range(state_blocks)
            ]
        )
        if self.training_weight_norm:
            for block in self.blocks:
                weight_norm(block.temporal, name="weight_ih_l0")
                weight_norm(block.temporal, name="weight_hh_l0")
                weight_norm(block.spectral.qkv)

        self.state_out = nn.Sequential(
            nn.Conv1d(state_channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels, eps=epsilon),
        )
        self.decoder = nn.ModuleList()
        for kernel in reversed(kernels[1:]):
            self.decoder.append(
                nn.Sequential(
                    nn.Conv1d(2 * channels, channels, 1, bias=False),
                    nn.BatchNorm1d(channels, eps=epsilon),
                    nn.SiLU(inplace=True),
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel,
                        padding=kernel // 2,
                        bias=False,
                    ),
                    nn.BatchNorm1d(channels, eps=epsilon),
                    nn.SiLU(inplace=True),
                )
            )



        final_projection: nn.Module
        if self.fastenhancer_codec:
            final_projection = _ScaledConvTranspose1d(
                channels,
                2,
                kernels[0],
                stride=stride,
                padding=(kernels[0] - stride) // 2,
            )
        else:
            final_projection = nn.ConvTranspose1d(
                channels,
                2,
                kernels[0],
                stride=stride,
                padding=(kernels[0] - stride) // 2,
            )
        self.output_projection = nn.Sequential(
            nn.Conv1d(2 * channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels, eps=epsilon),
            nn.SiLU(inplace=True),
            final_projection,
        )
        final = self.output_projection[-1]
        if self.identity_output_initialization:
            nn.init.normal_(final.weight, std=1.0e-3)
            nn.init.zeros_(final.bias)
            with torch.no_grad():
                final.bias[0] = 1.0






        if self.gram_observation:
            self.input[0] = nn.Conv1d(
                5,
                channels,
                kernels[0],
                stride=stride,
                padding=(kernels[0] - stride) // 2,
                bias=False,
            )
        elif self.simple_observation:
            self.input[0] = nn.Conv1d(
                3,
                channels,
                kernels[0],
                stride=stride,
                padding=(kernels[0] - stride) // 2,
                bias=False,
            )
        neighbour_kernel = torch.zeros(4, 2, 3)
        neighbour_kernel[0, 0, 0] = 1.0
        neighbour_kernel[1, 1, 0] = 1.0
        neighbour_kernel[2, 0, 2] = 1.0
        neighbour_kernel[3, 1, 2] = 1.0
        self.register_buffer("neighbour_kernel", neighbour_kernel, persistent=False)
        if self.gram_observation:
            self.phase_geometry = "gram_conv"
        elif self.simple_observation:
            self.phase_geometry = "power_compressed_complex_magnitude"
        else:
            self.phase_geometry = "raw_complex"
        self.gram_observation_folded = False
        self.deployed = False


        for stage, block in enumerate(self.blocks):
            bank = self.recurrent_weight_bank_assignment[stage]
            source = self.blocks[self.recurrent_weight_bank_sources[bank]]
            if source is block:
                continue
            _alias_weight(block.temporal, source.temporal, "weight_ih_l0")
            _alias_weight(block.temporal, source.temporal, "weight_hh_l0")

    def _observation_features(self, compressed: Tensor) -> Tensor:
        if self.simple_observation:
            batch, frequencies, frames, channels = compressed.shape
            complex_channels = compressed.permute(0, 2, 3, 1).reshape(
                batch * frames, channels, frequencies
            )
            magnitude = (
                compressed.square()
                .sum(-1)
                .clamp_min(self.epsilon**2)
                .sqrt()
                .permute(0, 2, 1)
                .reshape(batch * frames, 1, frequencies)
            )
            return torch.cat((complex_channels, magnitude), dim=1)
        if not self.gram_observation:
            batch, frequencies, frames, channels = compressed.shape
            return compressed.permute(0, 2, 3, 1).reshape(
                batch * frames, channels, frequencies
            )
        real = compressed[..., 0].permute(0, 2, 1)
        imaginary = compressed[..., 1].permute(0, 2, 1)
        batch, frames, frequencies = real.shape
        coordinates = torch.stack((real, imaginary), dim=2).reshape(batch * frames, 2, frequencies)
        neighbours = F.conv1d(coordinates, self.neighbour_kernel, padding=1)
        left_real, left_imaginary, right_real, right_imaginary = neighbours.unbind(1)
        real = coordinates[:, 0]
        imaginary = coordinates[:, 1]
        return torch.stack(
            (
                real.square() + imaginary.square(),
                left_real * real + left_imaginary * imaginary,
                left_imaginary * real - left_real * imaginary,
                right_real * real + right_imaginary * imaginary,
                right_imaginary * real - right_real * imaginary,
            ),
            dim=1,
        )

    def _state_to_codec(self, values: Tensor, batch: int, frames: int) -> Tensor:
        if self.fastenhancer_state_to_codec_layout:


            return values.permute(1, 0, 3, 2).reshape(
                batch * frames,
                self.state_channels,
                self.state_frequencies,
            )

        return (
            values.permute(1, 0, 3, 2)
            .reshape(
                batch * frames,
                self.state_frequencies,
                self.state_channels,
            )
            .transpose(1, 2)
        )

    def _mask_real(
        self,
        compressed: Tensor,
        states: tuple[Tensor | None, ...] | None = None,
    ) -> tuple[Tensor, tuple[Tensor, ...]]:
        if compressed.ndim != 4 or compressed.shape[-1] != 2:
            raise ValueError("compressed spectrum must be [batch,256,frames,2]")
        batch, frequencies, frames, _ = compressed.shape
        if frequencies != self.n_fft // 2:
            raise ValueError("Nyquist must be removed before Fiber")
        values = self.input(self._observation_features(compressed))
        skips = [values]
        for encoder in self.encoder:
            values = encoder(values)
            skips.append(values)
        values = self.state_in(F.linear(values, self.state_analysis))
        values = (
            values.reshape(batch, frames, self.state_channels, self.state_frequencies)
            .permute(1, 0, 3, 2)
            .contiguous()
        )
        if states is None:
            states = (None,) * len(self.blocks)
        if len(states) != len(self.blocks):
            raise ValueError("state tuple length does not match state_blocks")
        output_states = []
        for block, state in zip(self.blocks, states):
            values, state = block(values, state)
            output_states.append(state)
        values = self._state_to_codec(values, batch, frames)
        values = self.state_out(F.linear(values, self.state_synthesis))
        values = self.posterior_dropout(values)
        for decoder in self.decoder:
            values = decoder(torch.cat((values, skips.pop()), dim=1))
            values = self.decoder_dropout(values)
        values = self.output_projection(torch.cat((values, skips.pop()), dim=1))
        mask = values.reshape(batch, frames, 2, frequencies).permute(0, 3, 1, 2).contiguous()
        return mask, tuple(output_states)

    def _mask(
        self,
        compressed: Tensor,
        states: tuple[Tensor | None, ...] | None = None,
    ) -> tuple[Tensor, tuple[Tensor, ...]]:
        mask, states = self._mask_real(torch.view_as_real(compressed), states)
        return torch.view_as_complex(mask), states

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
        mask, _ = self._mask(compressed)
        enhanced_compressed = compressed * mask
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
            return_complex=False,
            length=noisy.shape[-1],
        )
        return waveform, torch.view_as_real(enhanced_compressed)

    def forward(self, waveform: Tensor) -> Tensor:
        return self.training_forward(waveform)[0]

    def initialize_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[Tensor, ...]:
        return tuple(
            torch.zeros(
                1,
                batch_size * block.temporal_state_batches,
                block.temporal_hidden_channels,
                device=device,
                dtype=dtype,
            )
            for block in self.blocks
        )

    initial_states = initialize_state

    def forward_frame(self, spectrum_frame: Tensor, *states: Tensor) -> tuple[Tensor, ...]:
        if spectrum_frame.ndim != 4 or spectrum_frame.shape[-1] != 2:
            raise ValueError("frame must be [batch,257,1,2]")
        body = spectrum_frame[:, : self.n_fft // 2]
        magnitude = body.square().sum(-1, keepdim=True).clamp_min(self.epsilon**2).sqrt()
        compressed = body * magnitude.pow(self.compression - 1.0)
        mask, output_states = self._mask_real(compressed, tuple(states) if states else None)
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

    @property
    def physical_recurrent_banks(self) -> int:
        signatures = {
            (
                id(block.temporal.parametrizations.weight_ih_l0.original0),
                id(block.temporal.parametrizations.weight_ih_l0.original1),
                id(block.temporal.parametrizations.weight_hh_l0.original0),
                id(block.temporal.parametrizations.weight_hh_l0.original1),
            )
            for block in self.blocks
        }
        return len(signatures)

    @torch.no_grad()
    def remove_weight_reparameterizations(self) -> None:
        if self.training_weight_norm:
            seen: set[int] = set()
            for block in self.blocks:
                temporal_id = id(block.temporal)
                if temporal_id not in seen:
                    remove_parametrizations(block.temporal, "weight_ih_l0")
                    remove_parametrizations(block.temporal, "weight_hh_l0")
                    seen.add(temporal_id)
                remove_parametrizations(block.spectral.qkv, "weight")

            for stage, block in enumerate(self.blocks):
                bank = self.recurrent_weight_bank_assignment[stage]
                source = self.blocks[self.recurrent_weight_bank_sources[bank]]
                if source is block:
                    continue
                block.temporal.weight_ih_l0 = source.temporal.weight_ih_l0
                block.temporal.weight_hh_l0 = source.temporal.weight_hh_l0
            for block in self.blocks:
                _fold_linear_batch_norm(
                    block.temporal_projection, block.temporal_norm.normalization
                )
                block.temporal_norm = nn.Identity()
                _fold_linear_batch_norm(
                    block.spectral_projection, block.spectral_norm.normalization
                )
                block.spectral_norm = nn.Identity()
            self.training_weight_norm = False
        if isinstance(self.input[0], _StridedConv1d):
            self.input[0] = _materialize_strided_conv(self.input[0])
        if isinstance(self.output_projection[-1], _ScaledConvTranspose1d):
            self.output_projection[-1] = _materialize_scaled_conv_transpose(
                self.output_projection[-1]
            )


__all__ = ["FiberModel"]
