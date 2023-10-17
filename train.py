import click
from os.path import join, dirname, abspath
import subprocess
import torch
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
import yaml
import datasets.datasets as datasets
import models.model as models
CUDA_LAUNCH_BLOCKING=1

@click.command()
### Add your options here
@click.option('--config',
              '-c',
              type=str,
              help='path to the config file (.yaml)',
              default=join(dirname(abspath(__file__)),'config/config.yaml'))
@click.option('--checkpoint',
              '-ckpt',
              type=str,
              help='path to checkpoint file (.ckpt) to resume training.',
              default=None)

def main(config, checkpoint):
    cfg = yaml.safe_load(open(config))
    # save the version of git we're using 
    #cfg['git_commit_version'] = str(subprocess.check_output(
    #        ['git', 'rev-parse', '--short', 'HEAD']).strip())

    # Load data and model
    data = datasets.UncuredFieldClouds(cfg)    
    model = models.BarlowTwinsModel(cfg)
    
    if checkpoint != None:
        model = model.load_from_checkpoint(checkpoint)
    

    # Add callbacks:
    #lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_saver = ModelCheckpoint(monitor='train:loss',
                                 save_top_k=5,
                                 filename=cfg['experiment']['id']+'_{epoch:02d}_{loss:.2f}',
                                 mode='min',
                                 save_last=True)


    tb_logger = pl_loggers.TensorBoardLogger('experiments/'+cfg['experiment']['id'],
                                             default_hp_metric=False)

    # Setup trainer
    trainer = Trainer(accumulate_grad_batches=16,
                      gpus=cfg['train']['n_gpus'],
                      logger=tb_logger,
                      resume_from_checkpoint=checkpoint,
                      max_epochs= cfg['train']['max_epoch']+1,
                      callbacks=[checkpoint_saver])
    # Train
    trainer.fit(model, data)

if __name__ == "__main__":
    main()
