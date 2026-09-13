from .fastenhancer_t import Model as FastEnhancerT
from .fastenhancer_b import Model as FastEnhancerB
from .fastenhancer_s import Model as FastEnhancerS
from .fiber_c import Model as FiberC
from .fiber_b import Model as FiberB
from .fiber_e import Model as FiberE


MODELS = {
    "fastenhancer_t": FastEnhancerT,
    "fastenhancer_b": FastEnhancerB,
    "fastenhancer_s": FastEnhancerS,
    "fiber_c": FiberC,
    "fiber_b": FiberB,
    "fiber_e": FiberE,
}


def build_model(name: str):
    return MODELS[name]()
