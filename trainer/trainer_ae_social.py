import os
import datetime
import torch
import torch.nn as nn
from models.model_encdec_social import model_encdec

from sddloader import *

torch.set_num_threads(5)

class Trainer:
    def __init__(self, config):
        
        self.name_test = str(datetime.datetime.now())[:10]
        self.folder_test = 'training/training_ae/' + self.name_test + '_' + config.info
        if not os.path.exists(self.folder_test):
            os.makedirs(self.folder_test)
        self.folder_test = self.folder_test + '/'
        

        # Creating dataset
        self.train_dataset = SocialDataset(set_name="train", b_size=config.train_b_size, t_tresh=config.time_thresh, d_tresh=config.dist_thresh)
        self.test_dataset = SocialDataset(set_name="test", b_size=config.test_b_size, t_tresh=config.time_thresh, d_tresh=config.dist_thresh)

        if torch.cuda.is_available(): torch.cuda.set_device(config.gpu)
        self.device = torch.device('cuda') if config.cuda else torch.device('cpu')

        self.settings = {
            "train_batch_size": config.train_b_size,
            "test_batch_size": config.test_b_size,
            "use_cuda": config.cuda,
            "dim_feature_tracklet": config.past_len * 2,
            "dim_feature_future": config.future_len * 2,
            "dim_embedding_key": config.dim_embedding_key,
            "past_len": config.past_len,
            "future_len": 1,
        }
        self.max_epochs = config.max_epochs

        # model
        self.mem_n2n = model_encdec(self.settings)
        

        # loss and optimization
        self.criterionLoss = nn.MSELoss()
        self.opt = torch.optim.Adam(self.mem_n2n.parameters(), lr=config.learning_rate)
        if config.cuda:
            self.criterionLoss = self.criterionLoss.cuda()
            self.mem_n2n = self.mem_n2n.cuda()
        self.start_epoch = 0
        self.config = config


    def print_model_param(self, model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("\033[1;31;40mTrainable/Total: {}/{}\033[0m".format(trainable_num, total_num))
        return 0
    
    
    def fit(self):
        
        self.print_model_param(self.mem_n2n)

            
        config = self.config
        # Training loop
        for epoch in range(self.start_epoch, config.max_epochs):

            print(' ----- Epoch: {}'.format(epoch))
            loss = self._train_single_epoch()
            print('Loss: {}'.format(loss))

            if epoch > 100 and (epoch + 1) % 10 == 0:
                dict_metrics_train = self.evaluate(self.train_dataset)

                dict_metrics_test = self.evaluate(self.test_dataset)

                # Save model checkpoint
                print('Train FDE_48s: {} ------ Train ADE: {} ------ Test FDE_48s: {} ------ Test ADE: {}'.format(dict_metrics_train['fde_48s'], \
                    dict_metrics_train['ade_48s'], dict_metrics_test['fde_48s'], dict_metrics_test['ade_48s']))
                
                torch.save(self.mem_n2n, self.folder_test + 'model_ae_epoch_' + str(epoch) + '_' + self.name_test)


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
                y = destination[:, -1:, :]

                abs_past = traj[:, :self.config.past_len, :]

                output, _ = self.mem_n2n(x, abs_past, seq_start_end, initial_pose, destination)
                output = output.data

                distances = torch.norm(output - y, dim=2)
                fde_48s += torch.sum(distances[:, -1])
                ade_48s += torch.sum(torch.mean(distances, dim=1))
                samples += distances.shape[0]

            dict_metrics['fde_48s'] = fde_48s / samples
            dict_metrics['ade_48s'] = ade_48s / samples

        return dict_metrics


    def _train_single_epoch(self):
        
        for i, (traj, mask, initial_pos,seq_start_end) \
            in enumerate(zip(self.train_dataset.trajectory_batches, self.train_dataset.mask_batches, self.train_dataset.initial_pos_batches, self.train_dataset.seq_start_end_batches)):
            traj, mask, initial_pos = torch.FloatTensor(traj).to(self.device), torch.FloatTensor(mask).to(self.device), torch.FloatTensor(initial_pos).to(self.device)
            # traj (B, T, 2)
            initial_pose = traj[:, 7, :] / 1000
            
            traj_norm = traj - traj[:, 7:8, :]
            x = traj_norm[:, :self.config.past_len, :]
            destination = traj_norm[:, -2:, :]
            y = destination[:, -1:, :]

            abs_past = traj[:, :self.config.past_len, :]

            self.opt.zero_grad()
            output, recon = self.mem_n2n(x, abs_past, seq_start_end, initial_pose, destination)

            self.opt.zero_grad()

            loss = self.criterionLoss(output, y) + self.criterionLoss(recon, x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.mem_n2n.parameters(), 1.0, norm_type=2)
            self.opt.step()


        return loss.item()
