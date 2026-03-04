import os
import misc as misc
import torch
from voxdet_core import Config
from voxdet_models import *
import pytorch_lightning as pl
from argparse import ArgumentParser
from LightningTools.pl_model import pl_model
from LightningTools.dataset_dm import DataModule
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.profilers import SimpleProfiler
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torch.distributed as dist



def is_main_process():
    if not dist.is_available() or not dist.is_initialized():
        return True     
    return dist.get_rank() == 0

def parse_config():
    parser = ArgumentParser()
    parser.add_argument('--config_path', default='./configs/semantic_kitti.py')
    parser.add_argument('--ckpt_path', default=None,
                        help='Checkpoint path for eval or training resume')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from last.ckpt in log_folder (or --ckpt_path)')
    parser.add_argument('--seed', type=int, default=7240, help='random seed point')
    parser.add_argument('--log_folder', default='logs/semantic_kitti')
    parser.add_argument('--save_path', default=None)
    parser.add_argument('--test_mapping', action='store_true')
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--log_every_n_steps', type=int, default=100)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--pretrain', action='store_true')

    args = parser.parse_args()
    cfg = Config.fromfile(args.config_path)

    cfg.update(vars(args))
    return args, cfg

if __name__ == '__main__':
    args, config = parse_config()
    log_folder = config['log_folder']
    misc.check_path(log_folder)

    misc.check_path(os.path.join(log_folder, 'tensorboard'))
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=log_folder,
        name='tensorboard'
    )

    # Wandb logger — resume the existing run when --resume is set
    model_type = config.model.get('type', 'VoxDet')
    wandb_resume = 'must' if config.get('resume', False) else None
    wandb_id = None
    if wandb_resume:
        # Try to recover the run ID from the last wandb run in this log folder
        wandb_dir = os.path.join(log_folder, 'wandb', 'latest-run')
        run_id_file = os.path.join(wandb_dir, 'run-id.txt') if os.path.isdir(wandb_dir) else ''
        if os.path.isfile(run_id_file):
            with open(run_id_file) as f:
                wandb_id = f.read().strip()
        else:
            # Fall back: don't force resume if no prior run found
            wandb_resume = 'allow'

    wandb_logger = pl_loggers.WandbLogger(
        project='VoxDet',
        name=f'{model_type}_{os.path.basename(log_folder)}',
        save_dir=log_folder,
        id=wandb_id,
        resume=wandb_resume,
        config={k: v for k, v in config.items()
                if isinstance(v, (int, float, str, bool, list, dict, type(None)))},
    )

    config.dump(os.path.join(log_folder, 'config.py'))
    profiler = SimpleProfiler(dirpath=log_folder, filename="profiler.txt")
    
    seed = config.seed
    pl.seed_everything(seed)
    num_gpu = torch.cuda.device_count()
    # num_gpu = 1
    model = pl_model(config)

    data_dm = DataModule(config)
    data_dm.setup()

    ckpt_dir = os.path.join(log_folder, 'checkpoints')
    misc.check_path(ckpt_dir)
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor='val/mIoU',
        mode='max',
        save_last=True,
        filename='best',
        save_on_train_epoch_end=False)
    # Save periodically so mid-epoch progress is not lost
    periodic_checkpoint = ModelCheckpoint(
        dirpath=ckpt_dir,
        every_n_train_steps=2000,
        save_top_k=-1,  # keep all periodic saves
        filename='step-{step}',
    )
    print("Is gpu available: ", torch.cuda.is_available())
    
    # OVVoxDet ensures all modules participate in the graph via dummy
    # forwards, so static_graph and find_unused_parameters are not needed.
    use_static_graph = False

    loggers = [tb_logger, wandb_logger]

    accumulate = config.get('accumulate_grad_batches', 1)

    if not config.eval:
        # TF32 for ~10-20% training speedup; eval uses full FP32 precision
        torch.set_float32_matmul_precision('medium')
        trainer = pl.Trainer(
            devices=[i for i in range(num_gpu)],
            strategy=DDPStrategy(
                accelerator='gpu',
                find_unused_parameters=False,
                static_graph=use_static_graph
            ),
            max_steps=config.training_steps,
            accumulate_grad_batches=accumulate,
            callbacks=[
                checkpoint_callback,
                periodic_checkpoint,
                LearningRateMonitor(logging_interval='step')
            ],
            logger=loggers,
            profiler=profiler,
            sync_batchnorm=True,
            log_every_n_steps=config['log_every_n_steps'],
            check_val_every_n_epoch=config['check_val_every_n_epoch']
        )
        # Resolve resume checkpoint path
        resume_ckpt = None
        if config.resume:
            if config.ckpt_path:
                resume_ckpt = config.ckpt_path
            else:
                # Auto-find last.ckpt in the log folder
                last_ckpt = os.path.join(ckpt_dir, 'last.ckpt')
                if os.path.isfile(last_ckpt):
                    resume_ckpt = last_ckpt
                else:
                    print(f'Warning: --resume set but no last.ckpt found at {last_ckpt}')
            if resume_ckpt:
                print(f'Resuming training from: {resume_ckpt}')

        trainer.fit(model=model, datamodule=data_dm, ckpt_path=resume_ckpt)
    else:
        if config['ckpt_path']:
            from voxdet_core import load_checkpoint
            load_checkpoint(model.model, config['ckpt_path'], map_location='cpu')
        trainer = pl.Trainer(
            devices=[i for i in range(num_gpu)],
            strategy=DDPStrategy(
                accelerator='gpu',
                find_unused_parameters=False
            ),
            logger=loggers,
            profiler=profiler
        )
        trainer.test(model=model, datamodule=data_dm)    
        
        