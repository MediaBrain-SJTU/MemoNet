import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.layer_utils import *


class model_encdec(nn.Module):
    
    def __init__(self, settings, pretrained_model=None):
        super(model_encdec, self).__init__()

        self.name_model = 'AIO_autoencoder'
        self.use_cuda = settings["use_cuda"]
        self.dim_embedding_key = 64 
        self.past_len = settings["past_len"]
        self.future_len = settings["future_len"]
        self.mode = settings["mode"]

        assert self.mode in ['intention', 'addressor_warm', 'addressor', 'trajectory'], 'WRONG MODE!'

        # LAYERS for different modes
        if self.mode == 'intention':
            self.abs_past_encoder = st_encoder()
            self.norm_past_encoder = st_encoder()
            self.norm_fut_encoder = st_encoder()

            self.res_past_encoder = st_encoder()
            self.social_pooling_X = NmpNet(
                embedding_dim=self.dim_embedding_key,
                h_dim=self.dim_embedding_key,
                mlp_dim=1024,
                bottleneck_dim=self.dim_embedding_key,
                activation='relu',
                batch_norm=False,
                nmp_layers=2
            )
            self.decoder = MLP(self.dim_embedding_key * 3, self.future_len * 2, hidden_size=(1024, 512, 1024))
            self.decoder_x = MLP(self.dim_embedding_key * 3, self.past_len * 2, hidden_size=(1024, 512, 1024))
            self.decoder_2 = MLP(self.dim_embedding_key * 3, self.future_len * 2, hidden_size=(1024, 512, 1024))
            self.decoder_2_x = MLP(self.dim_embedding_key * 3, self.past_len * 2, hidden_size=(1024, 512, 1024))
        else:
            self.abs_past_encoder = pretrained_model.abs_past_encoder
            self.norm_past_encoder = pretrained_model.norm_past_encoder
            self.norm_fut_encoder = pretrained_model.norm_fut_encoder
            self.res_past_encoder = pretrained_model.res_past_encoder
            self.social_pooling_X = pretrained_model.social_pooling_X
            self.decoder = pretrained_model.decoder
            self.decoder_x = pretrained_model.decoder_x
            self.decoder_2 = pretrained_model.decoder_2
            self.decoder_2_x = pretrained_model.decoder_2_x
            if self.mode == 'addressor_warm' or self.mode == 'addressor':
                for p in self.parameters():
                    p.requires_grad = False
                self.memory_past = torch.load('./training/saved_memory/sdd_social_filter_past.pt').cuda()
                self.memory_fut = torch.load('./training/saved_memory/sdd_social_filter_fut.pt').cuda()
                self.memory_dest = torch.load('./training/saved_memory/sdd_social_part_traj.pt').cuda()[:, -1]
                
                self.input_query_w = MLP(128, 128, (256, 256))
                self.past_memory_w = MLP(128, 128, (256, 256))
                if self.mode == 'addressor':
                    self.input_query_w = pretrained_model.input_query_w
                    self.past_memory_w = pretrained_model.past_memory_w
            else:
                self.memory_past = torch.load('./training/saved_memory/sdd_social_filter_past.pt').cuda()
                self.memory_fut = torch.load('./training/saved_memory/sdd_social_filter_fut.pt').cuda()
                self.memory_dest = torch.load('./training/saved_memory/sdd_social_part_traj.pt').cuda()[:, -1]
                
                self.input_query_w = pretrained_model.input_query_w
                self.past_memory_w = pretrained_model.past_memory_w
                for p in self.parameters():
                    p.requires_grad = False

                self.encoder_dest = MLP(input_dim = 2, output_dim = 16, hidden_size=(8, 16))
                self.traj_abs_past_encoder = st_encoder()
                self.interaction = NmpNet(
                    embedding_dim=self.dim_embedding_key,
                    h_dim=self.dim_embedding_key,
                    mlp_dim=1024,
                    bottleneck_dim=self.dim_embedding_key,
                    activation='relu',
                    batch_norm=False,
                    nmp_layers=2
                )
                self.num_decompose = 2
                self.decompose = nn.ModuleList([DecomposeBlock(self.past_len, self.future_len-1) for _ in range(self.num_decompose)])

        # activation function
        self.relu = nn.ReLU()


    def get_state_encoding(self, past, abs_past, seq_start_end, end_pose, future):
        norm_past_state = self.norm_past_encoder(past)
        abs_past_state = self.abs_past_encoder(abs_past)
        norm_fut_state = self.norm_fut_encoder(future)

        abs_past_state_social = self.social_pooling_X(abs_past_state, seq_start_end, end_pose)
        
        return norm_past_state, abs_past_state_social, norm_fut_state


    def decode_state_into_intention(self, past, norm_past_state, abs_past_state_social, norm_fut_state):
        # state concatenation and decoding
        input_fut = torch.cat((norm_past_state, abs_past_state_social, norm_fut_state), 1)
        prediction_y1 = self.decoder(input_fut).contiguous().view(-1, 1, 2)
        reconstruction_x1 = self.decoder_x(input_fut).contiguous().view(-1, self.past_len, 2)
        
        diff_past = past - reconstruction_x1 # B, T, 2
        diff_past_embed = self.res_past_encoder(diff_past) # B, F

        state_conc_diff = torch.cat((diff_past_embed, abs_past_state_social, norm_fut_state), 1)
        prediction_y2 = self.decoder_2(state_conc_diff).contiguous().view(-1, 1, 2)
        reconstruction_x2 = self.decoder_2_x(state_conc_diff).contiguous().view(-1, self.past_len, 2)
        
        prediction = prediction_y1 + prediction_y2
        reconstruction = reconstruction_x1 + reconstruction_x2
        
        return prediction, reconstruction

    
    def decode_state_into_sim_warm(self, norm_past_state, abs_past_state_social):
        state_past = torch.cat((norm_past_state, abs_past_state_social), dim=1)

        index_max, _ = self.get_memory_index(state_past, self.memory_past)

        memory_past = torch.Tensor().cuda()

        for i_track in range(200):
            i_ind = index_max[:, i_track]
            memory_past = torch.cat((memory_past, self.memory_past[i_ind].unsqueeze(1)), dim=1)
        
        state_past_selector = self.input_query_w(state_past)
        memory_past_selector = self.past_memory_w(memory_past)
        return state_past, state_past_selector, memory_past, memory_past_selector


    def decode_state_into_sim(self, norm_past_state, abs_past_state_social, future):
        state_past = torch.cat((norm_past_state, abs_past_state_social), dim=1)

        index_max, _ = self.get_memory_index(state_past, self.memory_past)

        memory_past = torch.Tensor().cuda()
        memory_fut = torch.Tensor().cuda()
        memory_destination =  torch.Tensor().cuda()

        for i_track in range(200):
            i_ind = index_max[:, i_track]
            memory_past = torch.cat((memory_past, self.memory_past[i_ind].unsqueeze(1)), dim=1)
            memory_fut = torch.cat((memory_fut, self.memory_fut[i_ind].unsqueeze(1)), dim=1)
            memory_destination = torch.cat((memory_destination, self.memory_dest[i_ind].unsqueeze(1)), dim=1)
        
        state_past_selector = self.input_query_w(state_past).unsqueeze(1)
        memory_past_selector = self.past_memory_w(memory_past)

        sample_memory_index, weight_read = self.get_memory_index_batch(state_past_selector, memory_past_selector)

        gt_destination = future[:, -1]
        distance = torch.sqrt(((gt_destination[:, None, :] - memory_destination[None, :, :])**2).sum(-1))
        return weight_read, distance.squeeze(0)


    def get_memory_index(self, state_past, memory_past):
        # state_past: batch_size, feature_size
        # memory_past: memory_size, feature_size
        past_normalized = F.normalize(memory_past, p=2, dim=1)
        state_normalized = F.normalize(state_past, p=2, dim=1)
        weight_read = torch.matmul(state_normalized, past_normalized.transpose(0, 1))
        _, index_max = torch.sort(weight_read, descending=True)
        return index_max, weight_read


    def get_memory_index_batch(self, state_past, memory_past):
        # state_past: batch_size, 1, feature_size
        # memory_past: batch_size, 300, feature_size
        past_normalized = F.normalize(memory_past, p=2, dim=2)
        state_normalized = F.normalize(state_past, p=2, dim=2)
        weight_read = torch.matmul(state_normalized, past_normalized.transpose(1, 2))
        weight_read = weight_read.squeeze()
        _, index_max = torch.sort(weight_read, descending=True)
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


    def generate_memory(self, train_dataset, filter_memory=False):
        
        self.memory_past = torch.Tensor().cuda()
        self.memory_fut = torch.Tensor().cuda()
        self.memory_dest = torch.Tensor().cuda()
        self.memory_start = torch.Tensor().cuda()
        self.traj = torch.Tensor().cuda()
        with torch.no_grad():
            for i, (traj, mask, initial_pos,seq_start_end) \
                in enumerate(zip(train_dataset.trajectory_batches, train_dataset.mask_batches, train_dataset.initial_pos_batches, train_dataset.seq_start_end_batches)):
                traj, mask, initial_pos = torch.FloatTensor(traj).cuda(), torch.FloatTensor(mask).cuda(), torch.FloatTensor(initial_pos).cuda()
                # traj (B, T, 2)
                initial_pose = traj[:, self.past_len-1, :] / 1000
                
                traj_norm = traj - traj[:, self.past_len-1:self.past_len, :]
                x = traj_norm[:, :self.past_len, :]
                destination = traj_norm[:, -2:, :]

                abs_past = traj[:, :self.past_len, :]

                state_past, state_past_social, state_fut = self.get_state_encoding(x, abs_past, seq_start_end, initial_pose, destination)
                
                state_past_total = torch.cat((state_past, state_past_social), dim=1)
                self.memory_past = torch.cat((self.memory_past, state_past_total), dim=0)
                self.memory_fut = torch.cat((self.memory_fut, state_fut), dim=0)
                self.memory_dest = torch.cat((self.memory_dest, destination[:, -1]), dim=0)
                self.memory_start = torch.cat((self.memory_start, x[:, 0]), dim=0)
                self.traj = torch.cat((self.traj, traj), dim=0)
        # print(self.memory_past.size())
        if filter_memory:
            index = [0]
            t_p = t_f = 0.5
            destination_memory = self.memory_dest[0:1]
            start_memory = self.memory_start[0:1]
            num_sample = self.memory_dest.shape[0]
            threshold_past = self.t_p
            threshold_futu = self.t_f
            for i in range(1, num_sample):
                memory_size = destination_memory.shape[0]
                distances = torch.norm(destination_memory - self.memory_dest[i].unsqueeze(0).repeat(memory_size, 1), dim=1)
                distances_start = torch.norm(start_memory - self.memory_start[i].unsqueeze(0).repeat(memory_size, 1), dim=1)

                mask_destination = torch.where(distances-threshold_past<t_f, torch.ones_like(distances), torch.zeros_like(distances))
                mask_start = torch.where(distances_start-threshold_futu<t_p, torch.ones_like(distances), torch.zeros_like(distances))

                mask = mask_destination + mask_start
                min_distance = torch.max(mask).item()
                if min_distance < 2:
                    index.append(i)
                    destination_memory = torch.cat((destination_memory, self.memory_dest[i].unsqueeze(0)), dim=0)
                    start_memory = torch.cat((start_memory, self.memory_start[i].unsqueeze(0)), dim=0)
            
            self.memory_past_after = self.memory_past[np.array(index)]
            self.memory_fut_after = self.memory_fut[np.array(index)]
            prefix_name = 'sdd_social_'+str(t_p)+'_'+str(t_f)+'_'

            torch.save(self.memory_fut_after, './training/saved_memory/{}{}_filter_fut.pt'.format(prefix_name, self.memory_past_after.shape[0]))
            torch.save(self.memory_past_after, './training/saved_memory/{}{}_filter_past.pt'.format(prefix_name, self.memory_past_after.shape[0]))
            torch.save(self.traj[np.array(index)], './training/saved_memory/{}{}_part_traj.pt'.format(prefix_name, self.memory_past_after.shape[0]))
           
            self.memory_past = self.memory_past_after.clone()
            self.memory_fut = self.memory_fut_after.clone()
        return 0


    def get_destination_from_memory(self, past, abs_past, seq_start_end, end_pose, return_abs=False):
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
        prediction = self.k_means(prediction.squeeze(2), ncluster=20, iter=10).unsqueeze(2)
        if return_abs:
            return prediction, abs_past_state_social
        return prediction


    def get_trajectory(self, past, abs_past, seq_start_end, end_pose):
        prediction = torch.Tensor().cuda()

        destination_prediction, abs_past_state_social = self.get_destination_from_memory(past, abs_past, seq_start_end, end_pose, return_abs=True)
        destination_prediction = destination_prediction.squeeze(2)
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


    def fixed_process_to_get_destination(self, past, abs_past, seq_start_end, end_pose, future):
        norm_past_state, abs_past_state_social, norm_fut_state = self.get_state_encoding(past, abs_past, seq_start_end, end_pose, future)
        prediction, _ = self.decode_state_into_intention(past, norm_past_state, abs_past_state_social, norm_fut_state)
        return prediction


    def forward(self, past, abs_past, seq_start_end, end_pose, future):
        norm_past_state, abs_past_state_social, norm_fut_state = self.get_state_encoding(past, abs_past, seq_start_end, end_pose, future)
        if self.mode == 'intention':
            prediction, reconstruction = self.decode_state_into_intention(past, norm_past_state, abs_past_state_social, norm_fut_state)
            return prediction, reconstruction
        elif self.mode == 'addressor_warm':
            state_past, state_past_selector, memory_past, memory_past_selector = self.decode_state_into_sim_warm(norm_past_state, abs_past_state_social)
            return state_past, state_past_selector, memory_past, memory_past_selector
        elif self.mode == 'addressor':
            weight_read, distance = self.decode_state_into_sim(norm_past_state, abs_past_state_social, future)
            return weight_read, distance
        else:
            destination_prediction = self.fixed_process_to_get_destination(past, abs_past, seq_start_end, end_pose, future)
            destination_prediction += torch.randn_like(destination_prediction)*5

            destination_feat = self.encoder_dest(destination_prediction.squeeze(1))
        
            abs_past_state = self.traj_abs_past_encoder(abs_past)
            abs_past_state_social = self.interaction(abs_past_state, seq_start_end, end_pose)
            
            state_conc = torch.cat((abs_past_state_social, destination_feat), dim=1)

            x_true = past.clone()
            x_hat = torch.zeros_like(x_true)
            batch_size = past.size(0)
            prediction = torch.zeros((batch_size, self.future_len-1, 2)).cuda()
            reconstruction = torch.zeros((batch_size, self.past_len, 2)).cuda()

            for i in range(self.num_decompose):
                x_hat, y_hat = self.decompose[i](x_true, x_hat, state_conc)
                prediction += y_hat
                reconstruction += x_hat
            
            for i in range(1, 12):
                prediction[:, i-1:i] += destination_prediction * i / 12 
            prediction = torch.cat((prediction, destination_prediction), dim=1)
            
            return prediction, reconstruction