import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layer_utils import *


class model_encdec(nn.Module):
    
    def __init__(self, settings, pretrained_model):
        super(model_encdec, self).__init__()

        self.name_model = 'autoencoder'
        self.use_cuda = settings["use_cuda"]
        self.dim_embedding_key = settings["dim_embedding_key"]*2 
        self.past_len = settings["past_len"]
        self.future_len = settings["future_len"]
        

        # temporal encoding
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

        for p in self.parameters():
            p.requires_grad=False

        # SDD
        self.memory_past = torch.load('./training/saved_memory/sdd_social_filter_past.pt').cuda()
        self.memory_fut = torch.load('./training/saved_memory/sdd_social_filter_fut.pt').cuda()
        self.memory_dest = torch.load('./training/saved_memory/sdd_social_part_traj.pt').cuda()[:, -1]
        


        self.input_query_w = MLP(128, 128, (256, 256))
        self.past_memory_w = MLP(128, 128, (256, 256))

        # activation function
        self.relu = nn.ReLU()


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
        weight_read = weight_read.squeeze()
        _, index_max = torch.sort(weight_read, descending=True)
        return index_max, weight_read
        
    def forward(self, past, abs_past, seq_start_end, end_pose, future):
      
        prediction = torch.Tensor()
        if self.use_cuda:
            prediction = prediction.cuda()

        # temporal encoding for past
        norm_past_state = self.norm_past_encoder(past)
        abs_past_state = self.abs_past_encoder(abs_past)
        norm_fut_state = self.norm_fut_encoder(future)

        abs_past_state_social = self.social_pooling_X(abs_past_state, seq_start_end, end_pose)

        state_past = torch.cat((norm_past_state, abs_past_state_social), dim=1)

        index_max, _ = self.get_memory_index(state_past, self.memory_past)


        memory_past = torch.Tensor().cuda()
        memory_fut = torch.Tensor().cuda()
        memory_destination =  torch.Tensor().cuda()

        for i_track in range(200):
            i_ind = index_max[:, i_track]
            memory_past = torch.cat((memory_past, self.memory_past[i_ind].unsqueeze(1)), dim=1)
            memory_fut = torch.cat((memory_fut, self.memory_fut[i_ind].unsqueeze(1)), dim=1)
        
        state_past_selector = self.input_query_w(state_past)
        memory_past_selector = self.past_memory_w(memory_past)
        if future is not None:
            return state_past, state_past_selector, memory_past, memory_past_selector

        for i_track in range(20):
            i_ind = index_max[:, i_track]
            feat_fut = self.memory_fut[i_ind]
            state_conc = torch.cat((state_past, feat_fut), 1)
            input_fut = state_conc
            prediction_y1 = self.decoder(input_fut).contiguous().view(-1, self.future_len, 2)
            reconstruction_x1 = self.decoder_x(input_fut).contiguous().view(-1, self.past_len, 2)
            
            diff_past = past - torch.transpose(reconstruction_x1, 1, 2) # B, 2, T
            diff_past_embed = self.relu(self.conv_past_2(diff_past))
            diff_past_embed = torch.transpose(diff_past_embed, 1, 2)

            _, state_past_diff = self.encoder_past_2(diff_past_embed)
            state_past_diff = state_past_diff.squeeze(0)

            state_conc_diff = torch.cat((state_past_diff, feat_fut), 1)
            prediction_y2 = self.decoder_2(state_conc_diff).contiguous().view(-1, self.future_len, 2)
            reconstruction_x2 = self.decoder_2_x(state_conc_diff).contiguous().view(-1, self.past_len, 2)
            
            prediction_single = prediction_y1 + prediction_y2
            prediction = torch.cat((prediction, prediction_single.unsqueeze(1)), dim=1)
        # prediction = self.k_means(prediction.squeeze(2), ncluster=20, iter=10).unsqueeze(2)
        return prediction
