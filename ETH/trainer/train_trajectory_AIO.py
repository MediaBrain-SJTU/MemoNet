import os
import math
import datetime
from random import sample

import torch
import torch.nn as nn

from models.model_train_trajectory import MemoNet

from data.dataloader import data_generator
from utils.config import Config
from utils.utils import prepare_seed, print_log

import time
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
		if self.cfg.model_encdec:
			self.MemoNet.model_encdec.load_state_dict(torch.load(self.cfg.model_encdec))
			print_log('Load model from {}.'.format(self.cfg.model_encdec), log=self.log)
		# loss
		self.criterionLoss = nn.MSELoss()

		trainable_layers = self.MemoNet.model_encdec.get_parameters(self.cfg.mode)
		self.opt = torch.optim.AdamW(trainable_layers.parameters(), lr=self.cfg.lr)
		self.iterations = 0
		if self.cfg.cuda:
			self.criterionLoss = self.criterionLoss.cuda()
			self.MemoNet = self.MemoNet.cuda()
		

	def fit(self):
		
		print_log('\n----------\nDataset: {}\nMode: {}\n----------\n'.format(self.cfg.dataset, self.cfg.mode), log=self.log)
		
		for epoch in range(self.cfg.num_epochs):

			loss = self._train_single_epoch()
			print_log('[{}] Epoch: {}/{}\tLoss: {:.6f}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), str(epoch), self.cfg.num_epochs, loss), log=self.log)

			if (epoch+1)%self.cfg.model_save_freq == 0:
				test_loss = self._evaluate(self.test_generator)
				print_log('------ Test loss: {}'.format(test_loss), log=self.log)
				cp_path = self.cfg.model_path % (epoch + 1)
				torch.save(self.MemoNet.model_encdec.state_dict(), cp_path)
		if self.cfg.mode == 'intention':
			self.generate_memory()


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
	
	
	def attention_loss(self, sim, distance):
		dis_mask = nn.MSELoss(reduction='sum')
		threshold_distance = 1
		mask = torch.where(distance>threshold_distance, torch.zeros_like(distance), torch.ones_like(distance))
		label_sim = (threshold_distance - distance) / threshold_distance
		label_sim = torch.where(label_sim<0, torch.zeros_like(label_sim), label_sim)
		loss = dis_mask(sim*mask, label_sim*mask) / (mask.sum()+1e-5)
		return loss


	def generate_memory(self):
		self.MemoNet.initial_memory()
		while not self.train_generator.is_epoch_end():
			data = self.train_generator()
			if data is not None:
				past = torch.stack(data['pre_motion_3D']).cuda()
				last_frame = past[:, -1:]
				future = torch.stack(data['fut_motion_3D']).cuda()
				future_part = future[:, -2:]
				past_normalized = past - last_frame
				fut_normalized = future_part - last_frame
				
				past_abs = past.unsqueeze(0).repeat(past.size(0), 1, 1, 1)
				past_centroid = past[:, -1:, :].unsqueeze(1)
				past_abs = past_abs - past_centroid

				# Data normalization
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

				self.MemoNet.add_memory(past_normalized, past_abs, end_pose, fut_normalized)
	

		self.MemoNet.filter_memory()
		
			
	def _train_single_epoch(self):
		self.train_generator.shuffle()
		count = loss_total = 0
		while not self.train_generator.is_epoch_end():
			data = self.train_generator()
			if data is not None:
				self.opt.zero_grad()
				past = torch.stack(data['pre_motion_3D']).cuda()
				last_frame = past[:, -1:]
				future = torch.stack(data['fut_motion_3D']).cuda()
				past_normalized = past - last_frame
				fut_normalized = future - last_frame
				
				past_abs = past.unsqueeze(0).repeat(past.size(0), 1, 1, 1)
				past_centroid = past[:, -1:, :].unsqueeze(1)
				past_abs = past_abs - past_centroid

				# Data normalization
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

				if self.cfg.mode == 'intention':
					prediction, reconstruction = self.MemoNet.reconstruct_destination(past_normalized, past_abs, end_pose, fut_normalized[:, -2:, :])
					loss = self.criterionLoss(prediction, fut_normalized[:, -1:, :]) + self.criterionLoss(reconstruction, past_normalized) 
				elif self.cfg.mode == 'addressor_warm':
					state_past, state_past_w, memory_past, past_memory_after, _ = self.MemoNet.get_attention(past_normalized, past_abs, end_pose, fut_normalized[:, -2:, :])
					loss = self.criterionLoss(state_past, state_past_w) + self.criterionLoss(memory_past, past_memory_after)
				elif self.cfg.mode == 'addressor':
					weight_read, distance = self.MemoNet.get_sim(past_normalized, past_abs, end_pose, fut_normalized[:, -2:, :])
					loss = self.attention_loss(weight_read, distance)
				else:
					output, recon = self.MemoNet.reconstruct_trajectory(past_normalized, past_abs, end_pose, fut_normalized[:, -2:, :])
					loss = self.criterionLoss(output, fut_normalized) + self.criterionLoss(recon, past_normalized)
				
				loss.backward()
				torch.nn.utils.clip_grad_norm_(self.MemoNet.parameters(), 1.0, norm_type=2)
				self.opt.step()

				loss_total += loss.item()
				count += 1

		return loss_total/count

		
	def _evaluate(self, generator):
		prepare_seed(self.cfg.seed)
		count = loss_total = 0
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

					if self.cfg.mode == 'intention':
						prediction, reconstruction = self.MemoNet.reconstruct_destination(past_normalized, past_abs, end_pose, fut_normalized[:, -2:, :])
						loss = self.criterionLoss(prediction, fut_normalized[:, -1:, :])
					elif self.cfg.mode == 'addressor_warm':
						state_past, state_past_w, memory_past, past_memory_after = self.MemoNet.get_attention(past_normalized, past_abs, end_pose, fut_normalized[:, -2:, :])
						loss = self.criterionLoss(state_past, state_past_w) + self.criterionLoss(memory_past, past_memory_after)
					elif self.cfg.mode == 'addressor':
						weight_read, distance = self.MemoNet.get_sim(past_normalized, past_abs, end_pose, fut_normalized[:, -2:, :])
						loss = self.attention_loss(weight_read, distance)
					else:
						output, recon = self.MemoNet.reconstruct_trajectory(past_normalized, past_abs, end_pose, fut_normalized[:, -2:, :])
						loss = self.criterionLoss(output, fut_normalized)
					loss_total += loss.item()
					count += 1

		return loss_total/count

