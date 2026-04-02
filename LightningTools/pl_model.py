import os
import torch
import numpy as np
import pytorch_lightning as pl
from .basemodel import LightningBaseModel
from .metric import SSCMetrics
from voxdet_core import build_model
from .utils import get_inv_map
from voxdet_core import load_checkpoint

class pl_model(LightningBaseModel):
    def __init__(
        self,
        config):
        super(pl_model, self).__init__(config)

        model_config = config['model']
        self.model = build_model(model_config)
        if 'load_from' in config:
            load_checkpoint(self.model, config['load_from'], map_location='cpu')
        
        self.num_class = config['num_class']
        self.class_names = config['class_names']
        
        self.train_metrics = SSCMetrics(config['num_class'])
        self.val_metrics = SSCMetrics(config['num_class'])
        self.test_metrics = SSCMetrics(config['num_class'])
        self.save_path = config['save_path']
        self.test_mapping = config['test_mapping']
        self.pretrain = config['pretrain']
    
    def forward(self, data_dict):
        return self.model(data_dict)
    
    def training_step(self, batch, batch_idx):
        output_dict = self.forward(batch)
        loss_dict = output_dict['losses']
        loss = 0
        for key, value in loss_dict.items():
            self.log(
                "train/"+key,
                value.detach(),
                on_epoch=True,
                sync_dist=False)
            loss += value

        self.log("train/loss",
            loss.detach(),
            on_epoch=True,
            sync_dist=False)

        # Compute train metrics only every 100 steps to avoid GPU stalls.
        # .cpu().numpy() forces a CUDA sync and the metric computation is
        # expensive (per-class np.where over 2M-element arrays).
        if not self.pretrain and (batch_idx % 100 == 0):
            pred = output_dict['pred'].detach().cpu().numpy()
            gt_occ = output_dict['gt_occ'].detach().cpu().numpy()

            self.train_metrics.add_batch(pred, gt_occ)

        return loss

    def on_before_optimizer_step(self, optimizer, *args, **kwargs):
        """Log gradient norms to detect exploding/vanishing gradients."""
        if self.global_step % self.trainer.log_every_n_steps != 0:
            return

        # Accumulate on GPU — only one .item() call at the very end
        device = next(self.model.parameters()).device
        total_sq = torch.zeros(1, device=device)
        module_sq = {}
        for name, module in self.model.named_children():
            msq = torch.zeros(1, device=device)
            for p in module.parameters():
                if p.grad is not None:
                    msq += p.grad.data.float().pow(2).sum()
            module_sq[name] = msq
            total_sq += msq

        # Single CUDA sync: move all norms to CPU at once
        keys = list(module_sq.keys())
        all_sq = torch.cat([total_sq] + [module_sq[k] for k in keys])
        all_norms = all_sq.sqrt().cpu()

        self.log("grad/total_norm", all_norms[0].item(), on_step=True, on_epoch=False)
        for i, name in enumerate(keys):
            if all_norms[i + 1] > 0:
                self.log(f"grad/{name}", all_norms[i + 1].item(), on_step=True, on_epoch=False)

    def validation_step(self, batch, batch_idx):
        output_dict = self.forward(batch)
        
        if not self.pretrain:
            pred = output_dict['pred'].detach().cpu().numpy()
            gt_occ = output_dict['gt_occ'].detach().cpu().numpy()

            self.val_metrics.add_batch(pred, gt_occ)

    def on_validation_epoch_end(self):
        metric_list = [("train", self.train_metrics), ("val", self.val_metrics)]
        # metric_list = [("val", self.val_metrics)]
        
        metrics_list = metric_list
        if self.trainer.is_global_zero:
            print('---------------Val----------------')
        for prefix, metric in metrics_list:
            stats = metric.get_stats()

            self.log("{}/mIoU".format(prefix), torch.tensor(stats["iou_ssc_mean"], dtype=torch.float32, device=self.device), sync_dist=True)
            self.log("{}/IoU".format(prefix), torch.tensor(stats["iou"], dtype=torch.float32, device=self.device), sync_dist=True)
            self.log("{}/Precision".format(prefix), torch.tensor(stats["precision"], dtype=torch.float32, device=self.device), sync_dist=True)
            self.log("{}/Recall".format(prefix), torch.tensor(stats["recall"], dtype=torch.float32, device=self.device), sync_dist=True)

            for name, iou in zip(self.class_names, stats['iou_ssc']):
                self.log("{}/IoU_{}".format(prefix,name), torch.tensor(iou, dtype=torch.float32, device=self.device), sync_dist=True)

            metric.reset()

    def test_step(self, batch, batch_idx):
        output_dict = self.forward(batch)

        pred = output_dict['pred'].detach().cpu().numpy()
        # gt_occ = None
        gt_occ = output_dict['gt_occ'] if 'gt_occ' in output_dict.keys() else None # for test_submit
        if gt_occ is not None:
            gt_occ = gt_occ.detach().cpu().numpy()
        else:
            gt_occ = None
            
        if self.save_path is not None:
            if self.test_mapping:
                inv_map = get_inv_map()
                output_voxels = inv_map[pred].astype(np.uint16)
            else:
                output_voxels = pred.astype(np.uint16)
            sequence_id = batch['img_metas']['sequence'][0]
            frame_id = batch['img_metas']['frame_id'][0]
            save_folder = "{}/sequences/{}/predictions".format(self.save_path, sequence_id)
            save_file = os.path.join(save_folder, "{}.label".format(frame_id))
            os.makedirs(save_folder, exist_ok=True)
            with open(save_file, 'wb') as f:
                output_voxels.tofile(f)
                print('\n save to {}'.format(save_file))

        if gt_occ is not None:
            self.test_metrics.add_batch(pred, gt_occ)
    # 
    def on_test_epoch_end(self):
        metric_list = [("test", self.test_metrics)]
        print('---------------Val----------------')
        metrics_list = metric_list
        for prefix, metric in metrics_list:
            stats = metric.get_stats()
            print('---------------IoU----------------')
            for name, iou in zip(self.class_names, stats['iou_ssc']):
                print(name + ":", iou)

            print('---------------TP----------------')
            for name, tp in zip(self.class_names, stats['tps_ssc']):
                print(name + ":", tp)

            print('---------------FP----------------')
            for name, fp in zip(self.class_names, stats['fps_ssc']):
                print(name + ":", fp)
            
            print('---------------FN----------------')
            for name, fn in zip(self.class_names, stats['fns_ssc']):
                print(name + ":", fn)
            self.log("{}/mIoU".format(prefix), torch.tensor(stats["iou_ssc_mean"], dtype=torch.float32, device=self.device), sync_dist=True)
            self.log("{}/IoU".format(prefix), torch.tensor(stats["iou"], dtype=torch.float32, device=self.device), sync_dist=True)
            self.log("{}/Precision".format(prefix), torch.tensor(stats["precision"], dtype=torch.float32, device=self.device), sync_dist=True)
            self.log("{}/Recall".format(prefix), torch.tensor(stats["recall"], dtype=torch.float32, device=self.device), sync_dist=True)

            for name, iou in zip(self.class_names, stats['iou_ssc']):
                self.log("{}/IoU_{}".format(prefix,name), torch.tensor(iou, dtype=torch.float32, device=self.device), sync_dist=True)
            metric.reset()

