from .fastenhancer_core import FastEnhancerModel


class FastEnhancerB(FastEnhancerModel):
    def __init__(self) -> None:
        super().__init__(
            channels=48, state_channels=36, state_frequencies=24, state_blocks=3, kernels=(8, 3, 3)
        )


Model = FastEnhancerB
__all__ = ["FastEnhancerB", "Model"]
