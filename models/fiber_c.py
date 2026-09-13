from .fiber_core import FiberModel


class Model(FiberModel):
    def __init__(self) -> None:
        super().__init__(
            channels=24,
            state_channels=20,
            state_frequencies=16,
            state_blocks=3,
            recurrent_weight_banks=3,
            recurrent_weight_bank_assignment=(0, 1, 2),
            group_size=4,
            alternate_groups=False,
            gram_observation=False,
            simple_observation=False,
            fastenhancer_filterbank=True,
            fastenhancer_codec=True,
            fastenhancer_state_to_codec_layout=True,
        )
        self.phase_geometry = "power_compressed_complex"
