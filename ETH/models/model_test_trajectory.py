import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.layer_utils import *

class model_encdec(nn.Module):
	
	def __init__(self):
		super(model_encdec, self).__init__()
		self.dim_embedding_key = 64 
		self.past_len = 8
		self.future_len = 1
		channel_in = 2
		channel_out = 16
		dim_kernel = 3
		input_gru = channel_out
		# LAYERS
		self.abs_past_encoder = st_encoder()
		self.norm_past_encoder = st_encoder()
		self.norm_fut_encoder = st_encoder()
		self.res_past_encoder = st_encoder()
		self.social_pooling_X = NmpNet_batch(nmp_layers=2)
		self.decoder = MLP(self.dim_embedding_key * 3, self.future_len * 2, hidden_size=(1024, 512, 1024))
		self.decoder_x = MLP(self.dim_embedding_key * 3, self.past_len * 2, hidden_size=(1024, 512, 1024))
		# self.decoder_x_abs = pretrained_model.decoder_x_abs
		self.decoder_2 = MLP(self.dim_embedding_key * 3, self.future_len * 2, hidden_size=(1024, 512, 1024))
		self.decoder_2_x = MLP(self.dim_embedding_key * 3, self.past_len * 2, hidden_size=(1024, 512, 1024))
		# self.decoder_2_x_abs = pretrained_model.decoder_2_x_abs

		self.input_query_w = MLP(128, 128, (256, 256))
		self.past_memory_w = MLP(128, 128, (256, 256))

		self.encoder_dest = MLP(input_dim = 2, output_dim = 16, hidden_size=(8, 16))
		self.traj_abs_past_encoder = st_encoder()
		self.interaction = NmpNet_batch(nmp_layers=2)
		self.num_decompose = 2
		self.decompose = nn.ModuleList([DecomposeBlock(self.past_len, 11) for _ in range(self.num_decompose)])



class MemoNet(nn.Module):
	
	def __init__(self, cfg):
		super(MemoNet, self).__init__()
		self.model_encdec = model_encdec()

		for p in self.parameters():
			p.requires_grad=False

		# time.sleep(60)
		self.memory_past = torch.load('{}/memory_past.pt'.format(cfg.memory_path), map_location=torch.device('cpu')).cuda()
		self.memory_fut = torch.load('{}/memory_fut.pt'.format(cfg.memory_path), map_location=torch.device('cpu')).cuda()
		self.num_decompose = 2
		# activation function
		self.relu = nn.ReLU()
		self.cfg = cfg

	def k_means(self, batch_x, ncluster=20, iter=10):
		"""return clustering ncluster of x.

		Args:
			x (Tensor): B, K, 2
			ncluster (int, optional): Number of clusters. Defaults to 20.
			iter (int, optional): Number of iteration to get the centroids. Defaults to 10.
		"""
		B, N, D = batch_x.size()
		batch_c = torch.Tensor().cuda()
		for i in range(B):
			x = batch_x[i]
			c = x[torch.randperm(N)[:ncluster]]
			for i in range(iter):
				a = ((x[:, None, :] - c[None, :, :])**2).sum(-1).argmin(1)
				c = torch.stack([x[a==k].mean(0) for k in range(ncluster)])
				nanix = torch.any(torch.isnan(c), dim=1)
				ndead = nanix.sum().item()
				c[nanix] = x[torch.randperm(N)[:ndead]]
			
			batch_c = torch.cat((batch_c, c.unsqueeze(0)), dim=0)
		return batch_c



	def get_memory_index(self, state_past, memory_past):
		past_normalized = F.normalize(memory_past, p=2, dim=1)
		state_normalized = F.normalize(state_past, p=2, dim=1)
		weight_read = torch.matmul(state_normalized, past_normalized.transpose(0, 1))
		_, index_max = torch.sort(weight_read, descending=True)
		# print('size of weight read:', weight_read.size())
		return index_max, weight_read


	def get_memory_index_batch(self, state_past, memory_past):
		# state_past: B, 1, F
		# memory_past: B, 300, F
		past_normalized = F.normalize(memory_past, p=2, dim=2)
		state_normalized = F.normalize(state_past, p=2, dim=2)
		weight_read = torch.matmul(state_normalized, past_normalized.transpose(1, 2))
		weight_read = weight_read.squeeze(1)
		_, index_max = torch.sort(weight_read, descending=True)
		return index_max, weight_read
	

	def get_destination(self, past, abs_past, end_pose):
		prediction = torch.Tensor().cuda()

		
		b1, b2, T, d = abs_past.size()

		# temporal encoding for past
		norm_past_state = self.model_encdec.norm_past_encoder(past)
		abs_past_state = self.model_encdec.abs_past_encoder(abs_past.contiguous().view(-1, T, d)).contiguous().view(b1, b2, -1)
		abs_past_state_social = self.model_encdec.social_pooling_X(abs_past_state, end_pose)
		abs_past_state_social = abs_past_state_social[torch.arange(0, b1), torch.arange(0, b1)]
		# abs_past_state_social = abs_past_state_social[:, 0]

		state_past = torch.cat((norm_past_state, abs_past_state_social), dim=1)

		index_max, _ = self.get_memory_index(state_past, self.memory_past)

		memory_past = torch.Tensor().cuda()
		memory_fut = torch.Tensor().cuda()

		for i_track in range(self.cfg.cosine_num):
			i_ind = index_max[:, i_track]
			memory_past = torch.cat((memory_past, self.memory_past[i_ind].unsqueeze(1)), dim=1)
			memory_fut = torch.cat((memory_fut, self.memory_fut[i_ind].unsqueeze(1)), dim=1)
		
		state_past_selector = self.model_encdec.input_query_w(state_past).unsqueeze(1)
		memory_past_selector = self.model_encdec.past_memory_w(memory_past)

		
		sample_memory_index, weight_read = self.get_memory_index_batch(state_past_selector, memory_past_selector)
		
		for i_track in range(self.cfg.selector_num):
			i_ind = sample_memory_index[:, i_track]
			feat_fut = memory_fut[torch.arange(0, len(i_ind)), i_ind]
			state_conc = torch.cat((state_past, feat_fut), 1)
			input_fut = state_conc
			prediction_y1 = self.model_encdec.decoder(input_fut).contiguous().view(-1, 1, 2)
			reconstruction_x1 = self.model_encdec.decoder_x(input_fut).contiguous().view(-1, 8, 2)
			
			diff_past = past - reconstruction_x1 # B, T, 2
			diff_past_embed = self.model_encdec.res_past_encoder(diff_past) # B, F

			state_conc_diff = torch.cat((diff_past_embed, abs_past_state_social, feat_fut), 1)
			prediction_y2 = self.model_encdec.decoder_2(state_conc_diff).contiguous().view(-1, 1, 2)
			# reconstruction_x2 = self.decoder_2_x(state_conc_diff).contiguous().view(-1, self.past_len, 2)
			
			prediction_single = prediction_y1 + prediction_y2
			prediction = torch.cat((prediction, prediction_single.unsqueeze(1)), dim=1)
		return prediction



	def forward(self, past, abs_past, end_pose):
		
		prediction = torch.Tensor().cuda()

		
		b1, b2, T, d = abs_past.size()

		abs_past_state = self.model_encdec.traj_abs_past_encoder(abs_past.contiguous().view(-1, T, d)).contiguous().view(b1, b2, -1)
		abs_past_state_social = self.model_encdec.interaction(abs_past_state, end_pose)
		abs_past_state_social = abs_past_state_social[torch.arange(0, b1), torch.arange(0, b1)]
		# N, 64 


		destination_prediction = self.get_destination(past, abs_past, end_pose).squeeze(2)
		# N, K, 2
		num_prediction = self.cfg.selector_num
		if not self.cfg.cluster_trajectory:
			num_prediction = 20
			destination_prediction = self.k_means(destination_prediction, ncluster=20, iter=10)
		for i in range(num_prediction):
			destination_feat = self.model_encdec.encoder_dest(destination_prediction[:, i])
			# N, 16
			
			state_conc = torch.cat((abs_past_state_social, destination_feat), dim=1)


			x_true = past.clone()
			x_hat = torch.zeros_like(x_true)
			batch_size = past.size(0)
			prediction_single = torch.zeros((batch_size, 11, 2)).cuda()
			reconstruction = torch.zeros((batch_size, 8, 2)).cuda()


			for decompose_i in range(self.num_decompose):
				x_hat, y_hat = self.model_encdec.decompose[decompose_i](x_true, x_hat, state_conc)
				prediction_single += y_hat
				reconstruction += x_hat
			
			if self.cfg.residual_prediction:
				for i_frame in range(1, 12):
					prediction_single[:, i_frame-1] += destination_prediction[:, i] * i_frame / 12 
			prediction_single = torch.cat((prediction_single, destination_prediction[:, i].unsqueeze(1)), dim=1)
			prediction = torch.cat((prediction, prediction_single.unsqueeze(1)), dim=1)
		
		if not self.cfg.cluster_trajectory:
			return prediction
		# B, K, T, 2
		# K = 245
		# destination_prediction: batch_size, k=245, 2
		destination_centroid = self.k_means(destination_prediction, ncluster=20, iter=10)
		destination_distance = ((destination_prediction[:, :, None, :] - destination_centroid[:, None, :, :])**2).sum(-1).argmin(2)
		
		# b1: batch_size
		# prediction: batch_size, K, T, 2
		prediction_final = torch.zeros((b1, 20, 12, 2)).cuda()
		for i in range(b1):
			prediction_final[i] = torch.stack([prediction[i, destination_distance[i]==k].mean(0) for k in range(20)])
		return prediction_final