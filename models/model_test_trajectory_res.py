import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class model_encdec(nn.Module):
    
    def __init__(self, settings, pretrained_model):
        super(model_encdec, self).__init__()

        self.name_model = 'autoencoder'
        self.use_cuda = settings["use_cuda"]
        self.dim_embedding_key = 64 
        self.past_len = settings["past_len"]
        self.future_len = settings["future_len"]
        
        # LAYERS
        self.abs_past_encoder = pretrained_model.abs_past_encoder
        self.norm_past_encoder = pretrained_model.norm_past_encoder
        self.norm_fut_encoder = pretrained_model.norm_fut_encoder
        self.res_past_encoder = pretrained_model.res_past_encoder
        self.social_pooling_X = pretrained_model.social_pooling_X
        self.decoder = pretrained_model.decoder
        self.decoder_x = pretrained_model.decoder_x
        # self.decoder_x_abs = pretrained_model.decoder_x_abs
        self.decoder_2 = pretrained_model.decoder_2
        self.decoder_2_x = pretrained_model.decoder_2_x
        # self.decoder_2_x_abs = pretrained_model.decoder_2_x_abs
        self.input_query_w = pretrained_model.input_query_w
        self.past_memory_w = pretrained_model.past_memory_w

        # MEMORY
        self.memory_past = torch.load('./training/saved_memory/sdd_social_filter_past.pt').cuda()
        self.memory_fut = torch.load('./training/saved_memory/sdd_social_filter_fut.pt').cuda()
        
        self.encoder_dest = pretrained_model.encoder_dest
        # self.encoder_dest = MLP(input_dim = 2, output_dim = 64, hidden_size=(64, 128))
        self.traj_abs_past_encoder = pretrained_model.traj_abs_past_encoder
        self.interaction = pretrained_model.interaction
        self.num_decompose = 2
        self.decompose = pretrained_model.decompose
        


        # activation function
        self.relu = nn.ReLU()

        for p in self.parameters():
            p.requires_grad = False

    def get_state_encoding(self, past, abs_past, seq_start_end, end_pose, future):
        norm_past_state = self.norm_past_encoder(past)
        abs_past_state = self.abs_past_encoder(abs_past)
        norm_fut_state = self.norm_fut_encoder(future)


        abs_past_state_social = self.social_pooling_X(abs_past_state, seq_start_end, end_pose)
        
        return norm_past_state, abs_past_state_social, norm_fut_state




    def get_memory_index(self, state_past, memory_past):
        past_normalized = F.normalize(memory_past, p=2, dim=1)
        state_normalized = F.normalize(state_past, p=2, dim=1)
        weight_read = torch.matmul(state_normalized, past_normalized.transpose(0, 1))
        _, index_max = torch.sort(weight_read, descending=True)
        # print('size of weight read:', weight_read.size())
        return index_max, weight_read



    
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


    def get_memory_index_batch(self, state_past, memory_past):
        # state_past: B, 1, F
        # memory_past: B, 200, F
        past_normalized = F.normalize(memory_past, p=2, dim=2)
        state_normalized = F.normalize(state_past, p=2, dim=2)
        weight_read = torch.matmul(state_normalized, past_normalized.transpose(1, 2))
        weight_read = weight_read.squeeze(1)
        _, index_max = torch.sort(weight_read, descending=True)
        return index_max, weight_read
        


    def fix_process_to_get_destination(self, past, abs_past, seq_start_end, end_pose):

        b1, T, d = abs_past.size()
        prediction = torch.Tensor()
        if self.use_cuda:
            prediction = prediction.cuda()

        # temporal encoding for past
        norm_past_state = self.norm_past_encoder(past)
        abs_past_state = self.abs_past_encoder(abs_past)
        abs_past_state_social = self.social_pooling_X(abs_past_state, seq_start_end, end_pose)
        
        state_past = torch.cat((norm_past_state, abs_past_state_social), dim=1)
        index_max, _ = self.get_memory_index(state_past, self.memory_past)

        memory_past = torch.Tensor().cuda()
        memory_fut = torch.Tensor().cuda()


        # state concatenation and decoding
        for i_track in range(200):
            i_ind = index_max[:, i_track]
            memory_past = torch.cat((memory_past, self.memory_past[i_ind].unsqueeze(1)), dim=1)
            memory_fut = torch.cat((memory_fut, self.memory_fut[i_ind].unsqueeze(1)), dim=1)
        
        state_past_selector = self.input_query_w(state_past).unsqueeze(1)
        memory_past_selector = self.past_memory_w(memory_past)

        # state_past_selector = state_past.unsqueeze(1)
        # memory_past_selector = memory_past.clone()

        sample_memory_index, weight_read = self.get_memory_index_batch(state_past_selector, memory_past_selector)
        
        
        for i_track in range(120):
            i_ind = sample_memory_index[:, i_track]
            feat_fut = memory_fut[torch.arange(0, len(i_ind)), i_ind]
            state_conc = torch.cat((state_past, feat_fut), 1)
            input_fut = state_conc
            prediction_y1 = self.decoder(input_fut).contiguous().view(-1, 1, 2)
            reconstruction_x1 = self.decoder_x(input_fut).contiguous().view(-1, self.past_len, 2)
            
            diff_past = past - reconstruction_x1 # B, T, 2
            diff_past_embed = self.res_past_encoder(diff_past) # B, F

            state_conc_diff = torch.cat((diff_past_embed, abs_past_state_social, feat_fut), 1)
            prediction_y2 = self.decoder_2(state_conc_diff).contiguous().view(-1, 1, 2)
            # reconstruction_x2 = self.decoder_2_x(state_conc_diff).contiguous().view(-1, self.past_len, 2)
            
            prediction_single = prediction_y1 + prediction_y2
            prediction = torch.cat((prediction, prediction_single.unsqueeze(1)), dim=1)

        prediction = self.k_means(prediction.squeeze(2), ncluster=20, iter=10)
        return prediction








    def forward(self, past, abs_past, seq_start_end, end_pose):
        prediction = torch.Tensor().cuda()

        abs_past_state = self.traj_abs_past_encoder(abs_past)
        abs_past_state_social = self.interaction(abs_past_state, seq_start_end, end_pose)
        # N, 64 
        destination_prediction = self.fix_process_to_get_destination(past, abs_past, seq_start_end, end_pose)
        # N, K, 2

        for i in range(20):
            destination_feat = self.encoder_dest(destination_prediction[:, i])
            # N, 16
            
            state_conc = torch.cat((abs_past_state_social, destination_feat), dim=1)


            x_true = past.clone()
            x_hat = torch.zeros_like(x_true)
            batch_size = past.size(0)
            prediction_single = torch.zeros((batch_size, 11, 2)).cuda()
            reconstruction = torch.zeros((batch_size, 8, 2)).cuda()


            for decompose_i in range(self.num_decompose):
                x_hat, y_hat = self.decompose[decompose_i](x_true, x_hat, state_conc)
                prediction_single += y_hat
                reconstruction += x_hat
            
            for i_frame in range(1, 12):
                prediction_single[:, i_frame-1] += destination_prediction[:, i] * i_frame / 12 
            prediction_single = torch.cat((prediction_single, destination_prediction[:, i].unsqueeze(1)), dim=1)
            prediction = torch.cat((prediction, prediction_single.unsqueeze(1)), dim=1)
        
        # B, K, T, 2
        return prediction