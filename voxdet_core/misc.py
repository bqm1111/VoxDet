import functools
import warnings
from itertools import repeat
from collections.abc import Iterable


def deprecated_api_warning(name_dict, cls_name=None):
    """Decorator to warn about deprecated API arguments."""
    def wrapper(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            for old_name, new_name in name_dict.items():
                if old_name in kwargs:
                    warnings.warn(
                        f'"{old_name}" is deprecated, use "{new_name}" instead',
                        DeprecationWarning, stacklevel=2)
                    if new_name not in kwargs:
                        kwargs[new_name] = kwargs.pop(old_name)
                    else:
                        kwargs.pop(old_name)
            return func(*args, **kwargs)
        return wrapped
    return wrapper


def to_2tuple(x):
    if isinstance(x, Iterable) and not isinstance(x, str):
        return tuple(x)
    return tuple(repeat(x, 2))
