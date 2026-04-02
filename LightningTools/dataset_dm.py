import torch
import pytorch_lightning as pl
from voxdet_core import build_dataset
from torch.utils.data.dataloader import DataLoader


# Keys whose tensors may have variable first-dimension across samples
# (e.g. different N_valid per frame) and therefore cannot be stacked.
_VARIABLE_LENGTH_KEYS = {
    'lseg_pixel_feat', 'lseg_confidence', 'valid_vox_indices',
}


def _collate_img_metas(metas_list):
    """Collate a list of img_metas dicts into a single dict.

    Tensor values are stacked; non-tensor values are kept as lists.
    This mirrors what torch's default_collate does with dicts.
    """
    if len(metas_list) == 1:
        return metas_list[0]
    keys = metas_list[0].keys()
    result = {}
    for k in keys:
        vals = [m[k] for m in metas_list]
        if isinstance(vals[0], torch.Tensor):
            try:
                result[k] = torch.stack(vals, dim=0)
            except RuntimeError:
                result[k] = vals
        else:
            result[k] = vals
    return result


def _ovo_collate_fn(batch):
    """Custom collate that keeps variable-length OVO tensors as lists."""
    elem = batch[0]
    result = {}
    for key in elem:
        values = [d[key] for d in batch]
        if key in _VARIABLE_LENGTH_KEYS:
            # Keep as a list — these have variable N_valid per sample
            result[key] = values
        elif key == 'img_metas':
            # Collate img_metas: stack tensors, keep non-tensors as lists
            result[key] = _collate_img_metas(values)
        elif values[0] is None:
            result[key] = None
        elif isinstance(values[0], torch.Tensor):
            try:
                result[key] = torch.stack(values, dim=0)
            except RuntimeError:
                # Fallback for any other variable-size tensors
                result[key] = values
        else:
            # Fallback: use default collation for non-tensor types
            result[key] = torch.utils.data.dataloader.default_collate(values)
    return result

class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        config      
    ):
        super().__init__()
        self.trainset_config = config.data.train
        self.testset_config = config.data.test
        self.valset_config = config.data.val

        self.train_dataloader_config = config.train_dataloader_config
        self.test_dataloader_config = config.test_dataloader_config
        self.val_dataloader_config = config.test_dataloader_config
        self.config = config
    
    def setup(self, stage=None):
        self.train_dataset = build_dataset(self.trainset_config)
        self.test_dataset = build_dataset(self.testset_config)
        self.val_dataset = build_dataset(self.valset_config)
        # Use custom collate when pipeline produces variable-length OVO tensors
        train_keys = set()
        for step in (self.trainset_config.get('pipeline', None) or []):
            for k in step.get('keys', []):
                train_keys.add(k)
        self._needs_ovo_collate = bool(train_keys & _VARIABLE_LENGTH_KEYS)
    
    def train_dataloader(self):
        num_workers = self.train_dataloader_config.num_workers
        collate = _ovo_collate_fn if self._needs_ovo_collate else None
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_dataloader_config.batch_size,
            drop_last=True,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
            collate_fn=collate)

    def val_dataloader(self):
        num_workers = self.val_dataloader_config.num_workers
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_dataloader_config.batch_size,
            drop_last=False,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=4 if num_workers > 0 else None)

    def test_dataloader(self):
        num_workers = self.test_dataloader_config.num_workers
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_dataloader_config.batch_size,
            drop_last=False,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=4 if num_workers > 0 else None)