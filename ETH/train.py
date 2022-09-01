import argparse

from trainer import train_trajectory_AIO as train_AIO

def parse_config():
	parser = argparse.ArgumentParser(description='[Train] MemoNet on ETH/UCY datasets')
	# Configuration for ETH/UCY dataset.
	parser.add_argument('--cfg', default='eth')
	parser.add_argument('--gpu', type=int, default=0)
	parser.add_argument('--tmp', action='store_true', default=False)
	
	parser.add_argument("--info", type=str, default='', help='Name of training/testing.')
	return parser.parse_args()


def main(config):
	t = train_AIO.Trainer(config)
	t.fit()


if __name__ == "__main__":
	config = parse_config()
	main(config)
