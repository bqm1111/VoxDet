import functools


def force_fp32(apply_to=None, out_fp16=False):
    """No-op decorator for native AMP compatibility.

    In the mmcv world this would cast inputs to fp32 before calling the
    function.  With native PyTorch AMP (autocast) this is unnecessary -
    autocast handles precision automatically.  We keep the decorator
    signature so existing call-sites don't break.
    """
    def wrapper(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapped
    return wrapper


def auto_fp16(apply_to=None, out_fp32=False):
    """No-op decorator for native AMP compatibility."""
    def wrapper(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapped
    return wrapper
