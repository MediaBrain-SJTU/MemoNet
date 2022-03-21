import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.layer_utils import *

class model_encdec(nn.Module):
    
    def __init__(self, settings):
        super(model_encdec, self).__init__()

        self.name_model = 'autoencoder'
        self.use_cuda = settings["use_cuda"]
        self.dim_embedding_key = 64 
        self.past_len = settings["past_len"]
        self.future_len = settings["future_len"]
        channel_in = 2
        channel_out = 16
        dim_kernel = 3
        input_gru = channel_out

        # encoders
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
        
        # activation function
        self.relu = nn.ReLU()

       
    def forward(self, past, abs_past, seq_start_end, end_pose, future):
        b1, T, d = abs_past.size()
        prediction = torch.Tensor()
        if self.use_cuda:
            prediction = prediction.cuda()

        # temporal encoding for past
        norm_past_state = self.norm_past_encoder(past)
        abs_past_state = self.abs_past_encoder(abs_past)
        norm_fut_state = self.norm_fut_encoder(future)


        abs_past_state_social = self.social_pooling_X(abs_past_state, seq_start_end, end_pose)
        
        # state concatenation and decoding
        input_fut = torch.cat((norm_past_state, abs_past_state_social, norm_fut_state), 1)
        prediction_y1 = self.decoder(input_fut).contiguous().view(-1, self.future_len, 2)
        reconstruction_x1 = self.decoder_x(input_fut).contiguous().view(-1, self.past_len, 2)
        # reconstruction_x1_abs = self.decoder_x_abs(input_fut).contiguous().view(-1, self.past_len, 2)
        
        diff_past = past - reconstruction_x1 # B, T, 2
        diff_past_embed = self.res_past_encoder(diff_past) # B, F

        state_conc_diff = torch.cat((diff_past_embed, abs_past_state_social, norm_fut_state), 1)
        prediction_y2 = self.decoder_2(state_conc_diff).contiguous().view(-1, self.future_len, 2)
        reconstruction_x2 = self.decoder_2_x(state_conc_diff).contiguous().view(-1, self.past_len, 2)
        # reconstruction_x2_abs = self.decoder_2_x_abs(state_conc_diff).contiguous().view(-1, self.past_len, 2)
        
        prediction = prediction_y1 + prediction_y2
        reconstruction = reconstruction_x1 + reconstruction_x2
        # reconstruction_abs = reconstruction_x1_abs + reconstruction_x2_abs

        return prediction, reconstruction
