import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

def encode_onehot(labels):
	classes = set(labels)
	classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
					enumerate(classes)}
	labels_onehot = np.array(list(map(classes_dict.get, labels)),
							 dtype=np.int32)
	return labels_onehot

def make_mlp(dim_list, activation='relu', batch_norm=True, dropout=0):
	layers = []
	for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
		layers.append(nn.Linear(dim_in, dim_out))
		if batch_norm:
			layers.append(nn.BatchNorm1d(dim_out))
		if activation == 'relu':
			layers.append(nn.ReLU())
		elif activation == 'leakyrelu':
			layers.append(nn.LeakyReLU())
		if dropout > 0:
			layers.append(nn.Dropout(p=dropout))
	return nn.Sequential(*layers)

def get_noise(shape, noise_type):
	if noise_type == 'gaussian':
		return torch.randn(*shape).cuda()
	elif noise_type == 'uniform':
		return torch.rand(*shape).sub_(0.5).mul_(2.0).cuda()
	raise ValueError('Unrecognized noise type "%s"' % noise_type)

class NmpNet_batch(nn.Module):
	"""Pooling module as proposed in our paper"""
	def __init__(
		self, embedding_dim=64, h_dim=64, mlp_dim=1024, bottleneck_dim=64,
		activation='relu', batch_norm=False, dropout=0.0, nmp_layers=2
	):
		super(NmpNet_batch, self).__init__()

		self.mlp_dim = mlp_dim
		self.h_dim = h_dim
		self.bottleneck_dim = bottleneck_dim
		self.embedding_dim = embedding_dim
		self.nmp_layers = nmp_layers
		self.batch_norm = batch_norm
		self.activation = activation

		mlp_pre_dim = embedding_dim + h_dim
		mlp_pre_pool_dims = [mlp_pre_dim, 512, bottleneck_dim]

		self.spatial_embedding = nn.Linear(2, embedding_dim)
		self.mlp_pre_pool = make_mlp(
			mlp_pre_pool_dims,
			activation=activation,
			batch_norm=batch_norm,
			dropout=dropout)

		self.nmp_mlp_start = make_mlp([h_dim*2+embedding_dim, 128, h_dim], activation=None, batch_norm=batch_norm, dropout=0.5)
		self.nmp_mlps = self.make_nmp_mlp()
		self.nmp_mlp_end = make_mlp([h_dim, 128, bottleneck_dim], activation=activation, batch_norm=batch_norm, dropout=0.5)
	
	def make_nmp_mlp(self):
		nmp_mlp = []
		for i in range(self.nmp_layers-1):
			mlp1 = make_mlp([self.h_dim, 128, self.h_dim], activation=None, batch_norm=self.batch_norm, dropout=0.5)
			mlp2 = make_mlp([self.h_dim*2, 128, self.h_dim], activation=None, batch_norm=self.batch_norm, dropout=0.5)
			nmp_mlp.append(mlp1)
			nmp_mlp.append(mlp2)
		nmp_mlp = nn.ModuleList(nmp_mlp)
		return nmp_mlp

	def edge2node(self, x, rel_rec, rel_send):
		# NOTE: Assumes that we have the same graph across all samples.
		incoming = torch.matmul(rel_rec.transpose(1, 2), x)
		return incoming / incoming.size(2)

	def node2edge(self, x, rel_rec, rel_send):
		# NOTE: Assumes that we have the same graph across all samples.
		# print(x.size())
		# print(rel_rec.size())
		# print(rel_send.size())
		receivers = torch.matmul(rel_rec, x)
		senders = torch.matmul(rel_send, x)
		edges = torch.cat([receivers, senders], dim=-1)
		return edges

	def init_adj(self, num_ped, batch_num):
		# rel_rec: [batch,N_edges, num_ped]
		# rel_send: [batch, N_edges, num_ped]
		# Generate off-diagonal interaction graph
		off_diag = np.ones([num_ped, num_ped])

		rel_rec = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
		rel_send = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
		rel_rec = torch.FloatTensor(rel_rec)
		rel_send = torch.FloatTensor(rel_send)

		rel_rec = rel_rec.cuda()
		rel_send = rel_send.cuda()
		rel_rec = rel_rec[None].repeat(batch_num,1,1)
		rel_send = rel_send[None].repeat(batch_num,1,1)

		return rel_rec, rel_send


	def forward(self, h_states, end_pos):
		"""
		Inputs:
		batch1: actor_dim, 0th -> 0th actor is ego
		batch2: scene_dim, all the other agents under ith ego agent
		- h_states: Tensor of shape (batch1, batch2, h_dim)
		- end_pos: Tensor of shape (batch1, batch2, 2)
		Output:
		- pool_h: Tensor of shape (batch1,batch2,bottleneck_dim)
		"""
		curr_hidden = h_states #(num_pred, num_pred, h_dim)
		curr_end_pos = end_pos   #(num_pred, num_pred, 2)
			
		# Repeat position -> P1, P2, P1, P2
		batch_num = h_states.shape[0]
		agent_num = h_states.shape[1]
		# if agent_num==1:
		#     return h_states
		curr_end_pos_1 = curr_end_pos[:,:,None,:]
			# Repeat position -> P1, P1, P2, P2
		curr_end_pos_2 = curr_end_pos[:,None,:,:]
		curr_rel_pos = (curr_end_pos_1 - curr_end_pos_2).view(batch_num,agent_num*agent_num,2)

		curr_rel_embedding = self.spatial_embedding(curr_rel_pos)

		# Neural Message Passing
		rel_rec, rel_send = self.init_adj(agent_num, batch_num)
		# iter 1
		edge_feat = self.node2edge(curr_hidden, rel_rec, rel_send) # [num_edge, h_dim*2]
		edge_feat = torch.cat([edge_feat, curr_rel_embedding], dim=-1)    # [num_edge, h_dim*2+embedding_dim]
		edge_feat = self.nmp_mlp_start(edge_feat)                      # [num_edge, h_dim]
			
		if self.nmp_layers <= 1:
			pass
		else:
			for nmp_l, nmp_mlp in enumerate(self.nmp_mlps):
				# print('-'*50)
				# print(edge_feat.size(), rel_rec.size(), rel_send.size())
				if nmp_l%2==0:
					node_feat = nmp_mlp(self.edge2node(edge_feat, rel_rec, rel_send)) # [num_ped, h_dim]
				else:    
					edge_feat = nmp_mlp(self.node2edge(node_feat, rel_rec, rel_send)) # [num_ped, h_dim] -> [num_edge, 2*h_dim] -> [num_edge, h_dim]
		
		node_feat = self.nmp_mlp_end(self.edge2node(edge_feat, rel_rec, rel_send))
		pool_h = node_feat

		return pool_h

'''MLP model'''
class MLP(nn.Module):
	def __init__(self, input_dim, output_dim, hidden_size=(1024, 512), activation='relu', discrim=False, dropout=-1):
		super(MLP, self).__init__()
		dims = []
		dims.append(input_dim)
		dims.extend(hidden_size)
		dims.append(output_dim)
		self.layers = nn.ModuleList()
		for i in range(len(dims)-1):
			self.layers.append(nn.Linear(dims[i], dims[i+1]))

		if activation == 'relu':
			self.activation = nn.ReLU()
		elif activation == 'sigmoid':
			self.activation = nn.Sigmoid()

		self.sigmoid = nn.Sigmoid() if discrim else None
		self.dropout = dropout

	def forward(self, x):
		for i in range(len(self.layers)):
			x = self.layers[i](x)
			if i != len(self.layers)-1:
				x = self.activation(x)
				if self.dropout != -1:
					x = nn.Dropout(min(0.1, self.dropout/3) if i == 1 else self.dropout)(x)
			elif self.sigmoid:
				x = self.sigmoid(x)
		return x


class st_encoder(nn.Module):
	def __init__(self):
		super().__init__()
		channel_in = 2
		channel_out = 16
		dim_kernel = 3
		self.dim_embedding_key = 64
		self.spatial_conv = nn.Conv1d(channel_in, channel_out, dim_kernel, stride=1, padding=1)
		self.temporal_encoder = nn.GRU(channel_out, self.dim_embedding_key, 1, batch_first=True)

		self.relu = nn.ReLU()

		self.reset_parameters()

	def reset_parameters(self):
		nn.init.kaiming_normal_(self.spatial_conv.weight)
		nn.init.kaiming_normal_(self.temporal_encoder.weight_ih_l0)
		nn.init.kaiming_normal_(self.temporal_encoder.weight_hh_l0)
		nn.init.zeros_(self.spatial_conv.bias)
		nn.init.zeros_(self.temporal_encoder.bias_ih_l0)
		nn.init.zeros_(self.temporal_encoder.bias_hh_l0)

	def forward(self, X):
		'''
		X: b, T, 2

		return: b, F
		'''
		X_t = torch.transpose(X, 1, 2)
		X_after_spatial = self.relu(self.spatial_conv(X_t))
		X_embed = torch.transpose(X_after_spatial, 1, 2)

		output_x, state_x = self.temporal_encoder(X_embed)
		state_x = state_x.squeeze(0)

		return state_x

class DecomposeBlock(nn.Module):
    '''
    Balance between reconstruction task and prediction task.
    '''
    def __init__(self, past_len, future_len):
        super(DecomposeBlock, self).__init__()
        # * HYPER PARAMETERS
        channel_in = 2
        channel_out = 16
        dim_kernel = 3
        dim_embedding_key = 64
        dest_embedding_key = 16
        self.past_len = past_len
        self.future_len = future_len

        self.conv_past = nn.Conv1d(channel_in, channel_out, dim_kernel, stride=1, padding=1)
        self.encoder_past = nn.GRU(channel_out, dim_embedding_key, 1, batch_first=True)
        
        self.decoder_y = MLP(dim_embedding_key * 2 + dest_embedding_key, future_len * 2, hidden_size=(1024, 512, 1024))
        self.decoder_x = MLP(dim_embedding_key * 2 + dest_embedding_key, past_len * 2, hidden_size=(1024, 512, 1024))

        self.relu = nn.ReLU()

        # kaiming initialization
        self.init_parameters()

    def init_parameters(self):
        nn.init.kaiming_normal_(self.conv_past.weight)
        nn.init.kaiming_normal_(self.encoder_past.weight_ih_l0)
        nn.init.kaiming_normal_(self.encoder_past.weight_hh_l0)

        nn.init.zeros_(self.conv_past.bias)
        nn.init.zeros_(self.encoder_past.bias_ih_l0)
        nn.init.zeros_(self.encoder_past.bias_hh_l0)


    def forward(self, x_true, x_hat, f):
        '''
        >>> Input:
            x_true: N, T_p, 2
            x_hat: N, T_p, 2
            f: N, f (128+16)

        >>> Output:
            x_hat_after: N, T_p, 2
            y_hat: n, T_f, 2
        '''
        x_ = x_true - x_hat
        x_ = torch.transpose(x_, 1, 2)
        
        past_embed = self.relu(self.conv_past(x_))
        past_embed = torch.transpose(past_embed, 1, 2)
        # N, T, F

        _, state_past = self.encoder_past(past_embed)
        state_past = state_past.squeeze(0)
        # N, F2

        input_feat = torch.cat((f, state_past), dim=1)

        x_hat_after = self.decoder_x(input_feat).contiguous().view(-1, self.past_len, 2)
        y_hat = self.decoder_y(input_feat).contiguous().view(-1, self.future_len, 2)
        
        return x_hat_after, y_hat



