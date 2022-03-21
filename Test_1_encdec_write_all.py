import argparse

from trainer import test_encdec_write_all as trainer_controllerMem
import numpy as np
import random
import torch

def prepare_seed(rand_seed):
	np.random.seed(rand_seed)
	random.seed(rand_seed)
	torch.manual_seed(rand_seed)
	torch.cuda.manual_seed_all(rand_seed)

def parse_config():
    parser = argparse.ArgumentParser(description='Test the trained models on SDD')

    # Configuration for training.
    parser.add_argument("--cuda", default=True)
    parser.add_argument("--learning_rate", type=int, default=0.001)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)


    # Configuration for SDD dataset.
    parser.add_argument("--data_scale", type=float, default=1)
    parser.add_argument("--data_scale_old", type=float, default=1.86)
    parser.add_argument("--train_b_size", type=int, default=512)
    parser.add_argument("--test_b_size", type=int, default=4096)
    parser.add_argument("--time_thresh", type=int, default=0)
    parser.add_argument("--dist_thresh", type=int, default=100)
    parser.add_argument("--past_len", type=int, default=8, help="length of past (in timesteps)")
    parser.add_argument("--future_len", type=int, default=12, help="length of future (in timesteps)")
    parser.add_argument("--dim_embedding_key", type=int, default=48)
    

    parser.add_argument("--info", type=str, default='', help='Name of training. '
                                                             'It will be used in tensorboard log and test folder')
    return parser.parse_args()


def main(config):
    for ae in range(109, 600, 10):
        config.model_ae = './training/training_ae/[your_file_name_here]/model_ae_epoch_' + str(ae) + '_2021-10-28'
        t = trainer_controllerMem.Trainer(config)
        print(config.model_ae)
        t.fit()
    

if __name__ == "__main__":
    config = parse_config()
    main(config)
