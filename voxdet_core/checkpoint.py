import os
import torch
import torch.nn as nn


def _remap_mmcv_keys(state_dict):
    """Remap old mmcv checkpoint keys to new voxdet_core naming conventions.

    Handles:
    - .bn. -> .norm. (BatchNorm layers)
    - .gn. -> .norm. (GroupNorm layers)
    - .conv_offset. -> .offset_conv. (DCN offset conv)
    - depth_conv.N.weight -> depth_conv.N.dcn.weight (DCN weight at parent level)
    - Drops num_batches_tracked keys (no counterpart in new model)
    """
    remapped = {}
    for k, v in state_dict.items():
        # Drop num_batches_tracked keys
        if k.endswith('.num_batches_tracked'):
            continue

        new_k = k

        # Remap BatchNorm: .bn. -> .norm.
        new_k = new_k.replace('.bn.', '.norm.')

        # Remap GroupNorm: .gn. -> .norm.
        new_k = new_k.replace('.gn.', '.norm.')

        # Remap DCN offset conv: .conv_offset. -> .offset_conv.
        new_k = new_k.replace('.conv_offset.', '.offset_conv.')

        # Remap DCN weight at parent level: depth_conv.N.weight -> depth_conv.N.dcn.weight
        # This handles the case where mmcv puts the DCN conv weight directly
        # under the parent module, but voxdet_core nests it under .dcn.
        if 'depth_conv.' in new_k and new_k.endswith('.weight'):
            # Check if this is a direct weight under a numbered layer (e.g. depth_conv.4.weight)
            parts = new_k.split('.')
            for i, part in enumerate(parts):
                if part == 'depth_conv' and i + 2 < len(parts):
                    next_part = parts[i + 1]
                    after_next = parts[i + 2]
                    if next_part.isdigit() and after_next == 'weight' and i + 2 == len(parts) - 1:
                        parts.insert(i + 2, 'dcn')
                        new_k = '.'.join(parts)
                        break

        remapped[new_k] = v
    return remapped


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

    # Remap old mmcv checkpoint keys to new voxdet_core naming
    new_state_dict = _remap_mmcv_keys(new_state_dict)

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
