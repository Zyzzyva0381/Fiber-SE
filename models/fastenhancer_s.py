from .fastenhancer_core import FastEnhancerModel


class FastEnhancerS(FastEnhancerModel):
    def __init__(self) -> None:
        super().__init__(
            channels=64,
            state_channels=48,
            state_frequencies=36,
            state_blocks=3,
            kernels=(8, 3, 3, 3),
        )


Model = FastEnhancerS
__all__ = ["FastEnhancerS", "Model"]
