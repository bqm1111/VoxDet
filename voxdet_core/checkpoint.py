import os
import torch
import torch.nn as nn


def load_checkpoint(model, filename, map_location=None, strict=False, logger=None):
    """Load checkpoint from a file.

    Handles various state_dict formats (bare dict, nested under 'state_dict',
    'model', etc.) and strips common prefixes like 'module.'.
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

    # Strip 'module.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        new_state_dict[k] = v

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
