import copy
import inspect


class Registry:
    """A registry to map strings to classes."""

    def __init__(self, name):
        self._name = name
        self._module_dict = {}

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key):
        return self._module_dict.get(key, None)

    def _register_module(self, module_class, module_name=None, force=False):
        if module_name is None:
            module_name = module_class.__name__
        if not force and module_name in self._module_dict:
            raise KeyError(f'{module_name} is already registered in {self._name}')
        self._module_dict[module_name] = module_class

    def register_module(self, name=None, force=False, module=None):
        if module is not None:
            self._register_module(module, name, force)
            return module

        # used as a decorator
        def _register(cls):
            self._register_module(cls, name, force)
            return cls

        return _register

    def build(self, cfg, **default_args):
        return build_from_cfg(cfg, self, default_args)


def build_from_cfg(cfg, registry, default_args=None):
    if not isinstance(cfg, dict):
        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
    if 'type' not in cfg:
        if default_args is None or 'type' not in default_args:
            raise KeyError('`cfg` or `default_args` must contain the key "type"')

    args = copy.deepcopy(cfg)
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)

    obj_type = args.pop('type')
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(f'{obj_type} is not in the {registry.name} registry')
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(f'type must be a str or class, but got {type(obj_type)}')

    return obj_cls(**args)


# All registries
DETECTORS = Registry('detector')
BACKBONES = Registry('backbone')
NECKS = Registry('neck')
HEADS = Registry('head')
ATTENTION = Registry('attention')
TRANSFORMER_LAYER = Registry('transformer_layer')
TRANSFORMER_LAYER_SEQUENCE = Registry('transformer_layer_sequence')
FEEDFORWARD_NETWORK = Registry('feedforward_network')
POSITIONAL_ENCODING = Registry('positional_encoding')
TRANSFORMER = Registry('transformer')
DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')


class _UnifiedModelsRegistry:
    """A composite registry that searches all model sub-registries.

    This mimics mm*'s single MODELS registry by looking up a type string
    across DETECTORS, BACKBONES, NECKS, HEADS, ATTENTION,
    TRANSFORMER_LAYER, TRANSFORMER_LAYER_SEQUENCE, FEEDFORWARD_NETWORK,
    POSITIONAL_ENCODING, and TRANSFORMER.
    """

    name = 'models'

    _sub_registries = [
        DETECTORS, BACKBONES, NECKS, HEADS, ATTENTION,
        TRANSFORMER_LAYER, TRANSFORMER_LAYER_SEQUENCE,
        FEEDFORWARD_NETWORK, POSITIONAL_ENCODING, TRANSFORMER,
    ]

    def get(self, key):
        for reg in self._sub_registries:
            cls = reg.get(key)
            if cls is not None:
                return cls
        return None

    def build(self, cfg, **default_args):
        return build_from_cfg(cfg, self, default_args)


MODELS = _UnifiedModelsRegistry()


# Builder functions
def build_backbone(cfg):
    return BACKBONES.build(cfg)


def build_neck(cfg):
    return NECKS.build(cfg)


def build_head(cfg):
    return HEADS.build(cfg)


def build_model(cfg):
    return DETECTORS.build(cfg)


def build_dataset(cfg):
    return DATASETS.build(cfg)


def build_attention(cfg):
    return ATTENTION.build(cfg)


def build_feedforward_network(cfg):
    return FEEDFORWARD_NETWORK.build(cfg)

def build_transformer_layer_sequence(cfg):
    return TRANSFORMER_LAYER_SEQUENCE.build(cfg)


def build_positional_encoding(cfg):
    return POSITIONAL_ENCODING.build(cfg)


def build_transformer(cfg):
    return TRANSFORMER.build(cfg)

