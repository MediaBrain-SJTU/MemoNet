import argparse
from trainer import trainer_ae_social as trainer_ae


def parse_config():
    parser = argparse.ArgumentParser(description='MemoNet for SDD')

    # Configuration for training.
    parser.add_argument("--cuda", default=True)
    parser.add_argument("--learning_rate", type=int, default=0.001)
    parser.add_argument("--max_epochs", type=int, default=600)
    parser.add_argument('--gpu', type=int, default=0)


    # Configuration for SDD dataset.
    parser.add_argument("--data_scale", type=float, default=1)
    parser.add_argument("--train_b_size", type=int, default=512)
    parser.add_argument("--test_b_size", type=int, default=4096)
    parser.add_argument("--time_thresh", type=int, default=0)
    parser.add_argument("--dist_thresh", type=int, default=100)
    parser.add_argument("--past_len", type=int, default=8, help="length of past (in timesteps)")
    parser.add_argument("--future_len", type=int, default=12, help="length of future (in timesteps)")
    parser.add_argument("--dim_embedding_key", type=int, default=48)
    

    parser.add_argument("--info", type=str, default='', help='Name of training. ')
    return parser.parse_args()


def main(config):
    t = trainer_ae.Trainer(config)
    print('[M] Start training autoencoder for SDD.')
    t.fit()


if __name__ == "__main__":
    config = parse_config()
    main(config)
