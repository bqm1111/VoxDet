import copy
import importlib.util
import os
import sys
import tempfile

class ConfigDict(dict):
    """A dictionary subclass that supports attribute-style access."""

    def __missing__(self, key):
        raise KeyError(key)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'ConfigDict' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(f"'ConfigDict' object has no attribute '{name}'")
    
    def __deepcopy__(self, memo):
        new = ConfigDict()
        for key, value in self.items():
            new[copy.deepcopy(key, memo)] = copy.deepcopy(value, memo)
        return new


    def copy(self):
        return copy.copy(self)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


def _dict_to_configdict(d):
    """Recursively convert a dict to ConfigDict."""
    if isinstance(d, dict) and not isinstance(d, ConfigDict):
        d = ConfigDict(d)
        for key, value in d.items():
            d[key] = _dict_to_configdict(value)
    elif isinstance(d, (list, tuple)):
        d = type(d)(_dict_to_configdict(v) for v in d)
    return d


class Config:
    """A config class that loads configuration from a Python file."""

    def __init__(self, cfg_dict=None, filename=None):
        if cfg_dict is None:
            cfg_dict = {}
        elif not isinstance(cfg_dict, dict):
            raise TypeError(f'cfg_dict must be a dict, but got {type(cfg_dict)}')

        for key, value in cfg_dict.items():
            if isinstance(value, dict) and not isinstance(value, ConfigDict):
                cfg_dict[key] = _dict_to_configdict(value)

        super().__setattr__('_cfg_dict', ConfigDict(cfg_dict))
        super().__setattr__('_filename', filename)

    @staticmethod
    def fromfile(filename):
        filename = os.path.abspath(os.path.expanduser(filename))
        if not os.path.isfile(filename):
            raise FileNotFoundError(f'config file {filename} does not exist')
        if not filename.endswith('.py'):
            raise IOError('Only .py config files are supported')

        spec = importlib.util.spec_from_file_location('_config_module', filename)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        cfg_dict = {}
        for key, value in mod.__dict__.items():
            if not key.startswith('__'):
                cfg_dict[key] = value

        return Config(cfg_dict, filename=filename)

    @property
    def filename(self):
        return self._filename

    def __getattr__(self, name):
        return getattr(self._cfg_dict, name)

    def __getitem__(self, name):
        return self._cfg_dict[name]

    def __setattr__(self, name, value):
        self._cfg_dict[name] = value

    def __setitem__(self, name, value):
        self._cfg_dict[name] = value

    def __contains__(self, key):
        return key in self._cfg_dict

    def __iter__(self):
        return iter(self._cfg_dict)

    def __len__(self):
        return len(self._cfg_dict)

    def __repr__(self):
        return f'Config(filename={self._filename}): {dict(self._cfg_dict)}'

    def update(self, d):
        self._cfg_dict.update(_dict_to_configdict(d))

    def get(self, key, default=None):
        return self._cfg_dict.get(key, default)

    def dump(self, filepath=None):
        """Dump config to a file or return as string."""
        lines = []
        for key, value in self._cfg_dict.items():
            lines.append(f'{key} = {repr(value)}')
        text = '\n'.join(lines)
        if filepath is not None:
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            with open(filepath, 'w') as f:
                f.write(text)
        return text

    def keys(self):
        return self._cfg_dict.keys()

    def values(self):
        return self._cfg_dict.values()

    def items(self):
        return self._cfg_dict.items()
