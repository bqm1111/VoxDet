class InstanceData:
    """Simple attribute container (replaces mmengine.structures.InstanceData)."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class PixelData:
    """Simple attribute container (replaces mmengine.structures.PixelData)."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
