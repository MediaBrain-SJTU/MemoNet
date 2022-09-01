import os
import math
import datetime
from random import sample

import torch
import torch.nn as nn

from models.model_test_trajectory import MemoNet

from data.dataloader import data_generator
from utils.config import Config
from utils.utils import prepare_seed, print_log

torch.set_num_threads(5)


class Trainer:
	def __init__(self, config):
		"""
		The Trainer class handles the training procedure for training the autoencoder.
		:param config: configuration parameters (see train_ae.py)
		"""
		
		self.cfg = Config(config.cfg, config.info, config.tmp, create_dirs=True)
		torch.set_default_dtype(torch.float32)
		if torch.cuda.is_available(): torch.cuda.set_device(config.gpu)

		self.log = open(os.path.join(self.cfg.log_dir, 'log.txt'), 'a+')
		self.train_generator = data_generator(self.cfg, self.log, split='train', phase='training')
		self.eval_generator = data_generator(self.cfg, self.log, split='val', phase='testing')
		self.test_generator = data_generator(self.cfg, self.log, split='test', phase='testing')

		self.max_epochs = self.cfg.num_epochs
		# model
		self.MemoNet = MemoNet(self.cfg)
		self.MemoNet.model_encdec.load_state_dict(torch.load(self.cfg.model_encdec))
		# loss
		self.criterionLoss = nn.MSELoss()

		self.opt = torch.optim.Adam(self.MemoNet.parameters(), lr=self.cfg.lr)
		self.iterations = 0
		if self.cfg.cuda:
			self.criterionLoss = self.criterionLoss.cuda()
			self.MemoNet = self.MemoNet.cuda()
		

	def fit(self):
		dict_metrics_test = self.evaluate(self.test_generator)
		print_log('------ Test FDE_48s: {} ------ Test ADE: {}'.format(dict_metrics_test['fde_48s'], dict_metrics_test['ade_48s']), log=self.log)
			 
	def rotate_traj(self, past, future, past_abs):
		past_diff = past[:, 0]
		past_theta = torch.atan(torch.div(past_diff[:, 1], past_diff[:, 0]+1e-5))
		past_theta = torch.where((past_diff[:, 0]<0), past_theta+math.pi, past_theta)

		rotate_matrix = torch.zeros((past_theta.size(0), 2, 2)).to(past_theta.device)
		rotate_matrix[:, 0, 0] = torch.cos(past_theta)
		rotate_matrix[:, 0, 1] = torch.sin(past_theta)
		rotate_matrix[:, 1, 0] = - torch.sin(past_theta)
		rotate_matrix[:, 1, 1] = torch.cos(past_theta)

		past_after = torch.matmul(rotate_matrix, past.transpose(1, 2)).transpose(1, 2)
		future_after = torch.matmul(rotate_matrix, future.transpose(1, 2)).transpose(1, 2)
		
		b1 = past_abs.size(0)
		b2 = past_abs.size(1)
		for i in range(b1):
			past_diff = (past_abs[i, 0, 0]-past_abs[i, 0, -1]).unsqueeze(0).repeat(b2, 1)
			past_theta = torch.atan(torch.div(past_diff[:, 1], past_diff[:, 0]+1e-5))
			past_theta = torch.where((past_diff[:, 0]<0), past_theta+math.pi, past_theta)

			rotate_matrix = torch.zeros((b2, 2, 2)).to(past_theta.device)
			rotate_matrix[:, 0, 0] = torch.cos(past_theta)
			rotate_matrix[:, 0, 1] = torch.sin(past_theta)
			rotate_matrix[:, 1, 0] = - torch.sin(past_theta)
			rotate_matrix[:, 1, 1] = torch.cos(past_theta)
			# print(past_abs.size())
			past_abs[i] = torch.matmul(rotate_matrix, past_abs[i].transpose(1, 2)).transpose(1, 2)
		# print('-'*50)
		# print(past_abs.size())
		return past_after, future_after, past_abs

	def evaluate(self, generator):
		prepare_seed(self.cfg.seed)
		ade_48s = fde_48s = 0
		samples = 0
		dict_metrics = {}
		with torch.no_grad():
			count = 0
			while not generator.is_epoch_end():
				
				data = generator()
				if data is not None:
					
					past = torch.stack(data['pre_motion_3D']).cuda()
					future = torch.stack(data['fut_motion_3D']).cuda()
					last_frame = past[:, -1:]
					past_normalized = past - last_frame
					fut_normalized = future - last_frame
					

					past_abs = past.unsqueeze(0).repeat(past.size(0), 1, 1, 1)
					past_centroid = past[:, -1:, :].unsqueeze(1)
					past_abs = past_abs - past_centroid

					scale = 1
					if self.cfg.scale.use:
						scale = torch.mean(torch.norm(past_normalized[:, 0], dim=1)) / 3
						if scale<self.cfg.scale.threshold:
							scale = 1
						else:
							if self.cfg.scale.type == 'divide':
								scale = scale / self.cfg.scale.large
							elif self.cfg.scale.type == 'minus':
								scale = scale - self.cfg.scale.large
						if self.cfg.scale.type=='constant':
							scale = self.cfg.scale.value
						past_normalized = past_normalized / scale
						past_abs = past_abs / scale

					if self.cfg.rotation:
						past_normalized, fut_normalized, past_abs = self.rotate_traj(past_normalized, fut_normalized, past_abs)
					end_pose = past_abs[:, :, -1]

					prediction = self.MemoNet(past_normalized, past_abs, end_pose)

					prediction = prediction.data * scale

					future_rep = fut_normalized.unsqueeze(1).repeat(1, 20, 1, 1)
					distances = torch.norm(prediction - future_rep, dim=3)
					distances = torch.where(torch.isnan(distances), torch.full_like(distances, 10), distances)
					# N, K, T

					mean_distances = torch.mean(distances[:, :, -1:], dim=2)
					mean_distances_ade = torch.mean(distances, dim=2)

					index_min = torch.argmin(mean_distances, dim=1)
					min_distances = distances[torch.arange(0, len(index_min)), index_min]

					index_min_ade = torch.argmin(mean_distances_ade, dim=1)
					min_distances_ade = distances[torch.arange(0, len(index_min_ade)), index_min_ade]

					fde_48s += torch.sum(min_distances[:, -1])
					ade_48s += torch.sum(torch.mean(min_distances_ade, dim=1))

					samples += distances.shape[0]
		dict_metrics['fde_48s'] = fde_48s / samples
		dict_metrics['ade_48s'] = ade_48s / samples

		return dict_metrics
