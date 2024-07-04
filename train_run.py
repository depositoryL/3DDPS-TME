import os
import torch
import argparse
import numpy as np

from Utils.logger import Logger
from DDPM.Engine.solver import Trainer
from Utils.data_utils import build_dataloader_sequence
from Utils.io import load_yaml_config, seed_everything, merge_opts_to_config, instantiate_from_config


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Training Script')
    parser.add_argument('--name', type=str, default=None)

    parser.add_argument('--config_file', type=str, default=None, 
                        help='path of config file')
    parser.add_argument('--output', type=str, default='OUTPUT', 
                        help='directory to save the results')
    parser.add_argument('--tensorboard', action='store_true', 
                        help='use tensorboard for logging')
    parser.add_argument('--clip', action='store_true', 
                        help='Delete Outlier.')
    parser.add_argument('--window', type=int, default=12, 
                        help='Length of sequnence.')

    # args for random

    parser.add_argument('--cudnn_deterministic', action='store_true', 
                        help='set cudnn.deterministic True')
    parser.add_argument('--seed', type=int, default=12345,
                        help='seed for initializing training.')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU id to use. If given, only the specific gpu will be'
                        ' used, and ddp will be disabled')
    
    # args for training
    parser.add_argument('--is_train', action='store_true', help='Train or Test.')
    parser.add_argument('--milestone', type=int, default=10)
    parser.add_argument('--is_sample', action='store_true', help='Sample or Estimate.')
    
    # args for modify config
    parser.add_argument('opts', help='Modify config options using the command-line',
                        default=None, nargs=argparse.REMAINDER)  

    args = parser.parse_args()
    args.save_dir = os.path.join(args.output, f'diffusion_{args.name}')

    if args.name == 'abilene':
        args.scale = 10 ** 9
        args.train_size = 15 * 7 * 288
        args.test_size = 1 * 7 * 288
        args.rm_filepath = './Data/abilene_rm.csv'
        args.tm_filepath = './Data/abilene_tm.csv'
    elif args.name == 'geant':
        args.scale = 10 ** 7
        args.train_size = 10 * 7 * 96
        args.test_size = 1 * 7 * 96
        args.rm_filepath = './Data/geant_rm.csv'
        args.tm_filepath = './Data/geant_tm.csv'
    else:
        raise ValueError(f'Unknown Network: {args.name}.')

    return args


def main():
    args = parse_args()

    if args.seed is not None:
        seed_everything(args.seed, args.cudnn_deterministic)
    
    config = load_yaml_config(args.config_file)
    config = merge_opts_to_config(config, args.opts)

    logger = Logger(args)
    logger.save_config(config)

    # ae_model = instantiate_from_config(config['model']['ae_model'])
    ddpm = instantiate_from_config(config['model']['diffusion'])

    train_loader, test_loader, rm, scale, shift = build_dataloader_sequence(args, config)
    ddpm.model.update(scale, shift)
    # ae_model.update(scale, shift)

    trainer = Trainer(ddpm, config, args, train_loader, logger)

    if args.is_train:
        trainer.train()
    elif args.is_sample:
        trainer.load(args.milestone)
        dataset = train_loader.dataset
        samples = trainer.sample(num=config['inference']['sample_size'], size_every=config['inference']['single_size'], size=[dataset.window, dataset.tm_dim])
        origin = dataset.traffic.cpu().numpy()
        # origin = dataset.scaler.inverse_transform(origin.reshape(origin.shape[0], -1)).reshape(origin.shape[0], samples.shape[1], samples.shape[-1])
        origin = origin.reshape(origin.shape[0], samples.shape[1], samples.shape[-1])
        np.save(os.path.join(args.save_dir, f'train_reals_{args.name}.npy'), origin)
        np.save(os.path.join(args.save_dir, f'ddpm_fake_{args.name}.npy'), samples)
    else:
        trainer.load(args.milestone)
        estimation, reals = trainer.estimate(test_loader, rm, learning_rate=config['inference']['guidance_lr'],
                                             use_em=config['inference']['use_em'])
        np.save(os.path.join(args.save_dir, f'test_reals_{args.name}.npy'), reals)
        np.save(os.path.join(args.save_dir, f'traffic_estimation_{args.name}.npy'), estimation)


if __name__ == '__main__':
    main()