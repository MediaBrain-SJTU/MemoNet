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
        self.t_p = settings["t_p"]
        self.t_f = settings["t_f"]
        

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


        # MEMORY
        self.memory_past = torch.Tensor().cuda()
        self.memory_fut = torch.Tensor().cuda()


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



    def write_all(self, train_dataset):
        
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
                initial_pose = traj[:, 7, :] / 1000
                
                traj_norm = traj - traj[:, 7:8, :]
                x = traj_norm[:, :8, :]
                destination = traj_norm[:, -2:, :]
                y = destination[:, -1:, :]

                abs_past = traj[:, :8, :]

                state_past, state_past_social, state_fut = self.get_state_encoding(x, abs_past, seq_start_end, initial_pose, destination)
                
                state_past_total = torch.cat((state_past, state_past_social), dim=1)
                self.memory_past = torch.cat((self.memory_past, state_past_total), dim=0)
                self.memory_fut = torch.cat((self.memory_fut, state_fut), dim=0)
                self.memory_dest = torch.cat((self.memory_dest, destination[:, -1]), dim=0)
                self.memory_start = torch.cat((self.memory_start, x[:, 0]), dim=0)
                self.traj = torch.cat((self.traj, traj), dim=0)
        
        if False:
            index = [0]
            destination_memory = self.memory_dest[0:1]
            start_memory = self.memory_start[0:1]
            num_sample = self.memory_dest.shape[0]
            threshold_past = self.t_p
            threshold_futu = self.t_f
            for i in range(1, num_sample):
                memory_size = destination_memory.shape[0]
                distances = torch.norm(destination_memory - self.memory_dest[i].unsqueeze(0).repeat(memory_size, 1), dim=1)
                distances_start = torch.norm(start_memory - self.memory_start[i].unsqueeze(0).repeat(memory_size, 1), dim=1)

                mask_destination = torch.where(distances-threshold_past<0, torch.ones_like(distances), torch.zeros_like(distances))
                mask_start = torch.where(distances_start-threshold_futu<0, torch.ones_like(distances), torch.zeros_like(distances))

                # mask_destination = torch.where(distances<0, torch.ones_like(distances), torch.zeros_like(distances))
                # mask_start = torch.where(distances_start<0, torch.ones_like(distances), torch.zeros_like(distances))
                
                mask = mask_destination + mask_start
                min_distance = torch.max(mask).item()
                if min_distance < 2:
                    index.append(i)
                    destination_memory = torch.cat((destination_memory, self.memory_dest[i].unsqueeze(0)), dim=0)
                    start_memory = torch.cat((start_memory, self.memory_start[i].unsqueeze(0)), dim=0)
            
            self.memory_past_after = self.memory_past[np.array(index)]
            self.memory_fut_after = self.memory_fut[np.array(index)]
            prefix_name = 'ablation/sdd_social_'+str(self.t_p)+'_'+str(self.t_f)+'_'

            torch.save(self.traj[np.array(index)], './training/saved_memory/{}{}_part_traj.pt'.format(prefix_name, self.memory_past_after.shape[0]))
            raise ValueError
            
            self.memory_past = self.memory_past_after.clone()
            self.memory_fut = self.memory_fut_after.clone()
        return 0


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

       
    def forward(self, past, abs_past, seq_start_end, end_pose):
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

        # state concatenation and decoding
        for i_track in range(200):
            i_ind = index_max[:, i_track]
            feat_fut = self.memory_fut[i_ind]

            input_fut = torch.cat((norm_past_state, abs_past_state_social, feat_fut), 1)
            prediction_y1 = self.decoder(input_fut).contiguous().view(-1, self.future_len, 2)
            reconstruction_x1 = self.decoder_x(input_fut).contiguous().view(-1, self.past_len, 2)
            # reconstruction_x1_abs = self.decoder_x_abs(input_fut).contiguous().view(-1, self.past_len, 2)
            
            diff_past = past - reconstruction_x1 # B, T, 2
            diff_past_embed = self.res_past_encoder(diff_past) # B, F

            state_conc_diff = torch.cat((diff_past_embed, abs_past_state_social, feat_fut), 1)
            prediction_y2 = self.decoder_2(state_conc_diff).contiguous().view(-1, self.future_len, 2)
            # reconstruction_x2 = self.decoder_2_x(state_conc_diff).contiguous().view(-1, self.past_len, 2)
            # reconstruction_x2_abs = self.decoder_2_x_abs(state_conc_diff).contiguous().view(-1, self.past_len, 2)
            
            prediction_single = prediction_y1 + prediction_y2
            # reconstruction = reconstruction_x1 + reconstruction_x2
            # reconstruction_abs = reconstruction_x1_abs + reconstruction_x2_abs
            prediction = torch.cat((prediction, prediction_single.unsqueeze(1)), dim=1)
        
        prediction = self.k_means(prediction.squeeze(2), ncluster=20, iter=10).unsqueeze(2)
        return prediction
