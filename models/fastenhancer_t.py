from .fastenhancer_core import FastEnhancerModel


class FastEnhancerT(FastEnhancerModel):
    def __init__(self) -> None:
        super().__init__(
            channels=24, state_channels=20, state_frequencies=16, state_blocks=2, kernels=(8, 3, 3)
        )


Model = FastEnhancerT
__all__ = ["FastEnhancerT", "Model"]
