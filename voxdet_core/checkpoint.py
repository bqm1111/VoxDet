import os
import torch
import torch.nn as nn


def _remap_keys(state_dict, model_keys):
    """Remap checkpoint keys to match the model's state_dict.

    Tries each key as-is first. If it doesn't match, applies a series of
    candidate transformations and picks the first one that matches a model key.

    Transformations tried (in order):
    - .bn. -> .norm.
    - .gn. -> .norm.
    - .conv_offset. -> .offset_conv.
    - depth_conv.N.weight -> depth_conv.N.dcn.weight
    - depth_conv.N.bias -> depth_conv.N.dcn.bias
    """
    remapped = {}
    for k, v in state_dict.items():
        # Skip num_batches_tracked — they are not essential and the model
        # initialises them to zero anyway.
        if k.endswith('.num_batches_tracked'):
            continue

        # Fast path: key already matches the model
        if k in model_keys:
            remapped[k] = v
            continue

        new_k = k

        # Try .bn. -> .norm.
        if '.bn.' in new_k:
            new_k = new_k.replace('.bn.', '.norm.')
        # Try .gn. -> .norm.
        if '.gn.' in new_k:
            new_k = new_k.replace('.gn.', '.norm.')
        # Try .conv_offset. -> .offset_conv.
        if '.conv_offset.' in new_k:
            new_k = new_k.replace('.conv_offset.', '.offset_conv.')
        # Try depth_conv.N.weight/bias -> depth_conv.N.dcn.weight/bias
        if 'depth_conv.' in new_k and (new_k.endswith('.weight') or new_k.endswith('.bias')):
            parts = new_k.split('.')
            for i, part in enumerate(parts):
                if part == 'depth_conv' and i + 2 < len(parts):
                    next_part = parts[i + 1]
                    after_next = parts[i + 2]
                    if next_part.isdigit() and after_next in ('weight', 'bias') and i + 2 == len(parts) - 1:
                        parts.insert(i + 2, 'dcn')
                        new_k = '.'.join(parts)
                        break

        remapped[new_k] = v
    return remapped


def load_checkpoint(model, filename, map_location=None, strict=False, logger=None):
    """Load checkpoint from a file.

    Handles various state_dict formats (bare dict, nested under 'state_dict',
    'model', etc.) and strips common prefixes like 'module.' and 'model.'
    (PyTorch Lightning wraps models under self.model).
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(f'No checkpoint found at {filename}')

    checkpoint = torch.load(filename, map_location=map_location, weights_only=False)

    # Extract state_dict from checkpoint
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Strip common prefixes
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        if k.startswith('model.'):
            k = k[6:]
        new_state_dict[k] = v

    # Remap checkpoint keys to match the model, guided by the model's own keys
    model_keys = set(model.state_dict().keys())
    new_state_dict = _remap_keys(new_state_dict, model_keys)

    # Load with non-strict by default to handle partial loads
    missing, unexpected = model.load_state_dict(new_state_dict, strict=strict)
    if logger:
        if missing:
            logger.warning(f'Missing keys: {missing}')
        if unexpected:
            logger.warning(f'Unexpected keys: {unexpected}')
    else:
        if missing:
            print(f'Warning: Missing keys in checkpoint: {len(missing)} keys')
        if unexpected:
            print(f'Warning: Unexpected keys in checkpoint: {len(unexpected)} keys')

    return checkpoint


class CheckpointLoader:
    """Simple checkpoint loader compatible with mmengine's interface."""

    @staticmethod
    def load_checkpoint(filename, map_location=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError(f'No checkpoint found at {filename}')
        return torch.load(filename, map_location=map_location, weights_only=False)
