import os
import datetime
import torch
import torch.nn as nn
from models.model_test_trajectory_res import model_encdec

from sddloader import *

torch.set_num_threads(5)

class Trainer:
    def __init__(self, config):

        # test folder creating
        self.name_test = str(datetime.datetime.now())[:10]
        self.folder_test = 'testing/' + self.name_test + '_' + config.info
        if not os.path.exists(self.folder_test):
            os.makedirs(self.folder_test)
        self.folder_test = self.folder_test + '/'
 
       
        self.test_dataset = SocialDataset(set_name="test", b_size=config.test_b_size, t_tresh=config.time_thresh, d_tresh=config.dist_thresh)

       
        if torch.cuda.is_available(): torch.cuda.set_device(config.gpu)

        self.settings = {
            "train_batch_size": config.train_b_size,
            "test_batch_size": config.test_b_size,
            "use_cuda": config.cuda,
            "dim_feature_tracklet": config.past_len * 2,
            "dim_feature_future": config.future_len * 2,
            "dim_embedding_key": config.dim_embedding_key,
            "past_len": config.past_len,
            "future_len": 12,
        }

        # model
        self.model_ae = torch.load(config.model_ae, map_location=torch.device('cpu')).cuda()
        self.mem_n2n = model_encdec(self.settings, self.model_ae)
        

        if config.cuda:
            self.mem_n2n = self.mem_n2n.cuda()
        self.start_epoch = 0
        self.config = config

        self.device = torch.device('cuda') if config.cuda else torch.device('cpu')

        


    def print_model_param(self, model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("\033[1;31;40mTrainable/Total: {}/{}\033[0m".format(trainable_num, total_num))
        return 0
    
    
    def fit(self):
        
        dict_metrics_test = self.evaluate(self.test_dataset)
        print('Test FDE_48s: {} ------ Test ADE: {}'.format(dict_metrics_test['fde_48s'], dict_metrics_test['ade_48s']))
        print('-'*100)
        

    def evaluate(self, dataset):
        
        ade_48s = fde_48s = 0
        samples = 0
        dict_metrics = {}

        with torch.no_grad():
            for i, (traj, mask, initial_pos,seq_start_end) \
                in enumerate(zip(dataset.trajectory_batches, dataset.mask_batches, dataset.initial_pos_batches, dataset.seq_start_end_batches)):
                traj, mask, initial_pos = torch.FloatTensor(traj).to(self.device), torch.FloatTensor(mask).to(self.device), torch.FloatTensor(initial_pos).to(self.device)
                # traj (B, T, 2)
                initial_pose = traj[:, 7, :] / 1000
                
                traj_norm = traj - traj[:, 7:8, :]
                x = traj_norm[:, :self.config.past_len, :]
                destination = traj_norm[:, -2:, :]
                

                abs_past = traj[:, :self.config.past_len, :]
                
                output = self.mem_n2n(x, abs_past, seq_start_end, initial_pose)
                output = output.data
                # B, K, t, 2

                future_rep = traj_norm[:, 8:, :].unsqueeze(1).repeat(1, 20, 1, 1)
                distances = torch.norm(output - future_rep, dim=3)
                mean_distances = torch.mean(distances[:, :, -1:], dim=2)
                index_min = torch.argmin(mean_distances, dim=1)
                min_distances = distances[torch.arange(0, len(index_min)), index_min]

                fde_48s += torch.sum(min_distances[:, -1])
                ade_48s += torch.sum(torch.mean(min_distances, dim=1))
                samples += distances.shape[0]


            dict_metrics['fde_48s'] = fde_48s / samples
            dict_metrics['ade_48s'] = ade_48s / samples

        return dict_metrics
