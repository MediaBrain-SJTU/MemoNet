import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.layer_utils import *



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

        for p in self.parameters():
            p.requires_grad = False


        self.encoder_dest = MLP(input_dim = 2, output_dim = 16, hidden_size=(8, 16))
        # self.encoder_dest = MLP(input_dim = 2, output_dim = 64, hidden_size=(64, 128))
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



    def fixed_process_to_get_destination(self, past, abs_past, seq_start_end, end_pose, future):
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
        prediction_y1 = self.decoder(input_fut).contiguous().view(-1, 1, 2)
        reconstruction_x1 = self.decoder_x(input_fut).contiguous().view(-1, self.past_len, 2)
        # reconstruction_x1_abs = self.decoder_x_abs(input_fut).contiguous().view(-1, self.past_len, 2)
        
        diff_past = past - reconstruction_x1 # B, T, 2
        diff_past_embed = self.res_past_encoder(diff_past) # B, F

        state_conc_diff = torch.cat((diff_past_embed, abs_past_state_social, norm_fut_state), 1)
        prediction_y2 = self.decoder_2(state_conc_diff).contiguous().view(-1, 1, 2)
        # reconstruction_x2 = self.decoder_2_x(state_conc_diff).contiguous().view(-1, self.past_len, 2)
        # reconstruction_x2_abs = self.decoder_2_x_abs(state_conc_diff).contiguous().view(-1, self.past_len, 2)
        
        prediction = prediction_y1 + prediction_y2
        # reconstruction = reconstruction_x1 + reconstruction_x2
        # reconstruction_abs = reconstruction_x1_abs + reconstruction_x2_abs
        return prediction


    def forward(self, past, abs_past, seq_start_end, end_pose, future):


        destination_prediction = self.fixed_process_to_get_destination(past, abs_past, seq_start_end, end_pose, future)
        # N, 1, 2

        destination_prediction += torch.randn_like(destination_prediction)*5

        destination_feat = self.encoder_dest(destination_prediction.squeeze(1))
        # N, 16

        abs_past_state = self.traj_abs_past_encoder(abs_past)
        abs_past_state_social = self.interaction(abs_past_state, seq_start_end, end_pose)
        # N, 64 

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
        
        # for i in range(1, 12):
        #     prediction[:, i-1:i] += destination_prediction * i / 12 
        prediction = torch.cat((prediction, destination_prediction), dim=1)
        

        return prediction, reconstruction