import os
import datetime
import torch
import torch.nn as nn
from models.model_AIO import model_encdec
from trainer.evaluations import *

from sddloader import *


torch.set_num_threads(5)

class Trainer:
    def __init__(self, config):
        """
        The Trainer class handles the training procedure for training the autoencoder.
        :param config: configuration parameters (see train_ae.py)
        """

        # test folder creating
        self.name_test = str(datetime.datetime.now())[:10]
        self.folder_test = 'training/training_' + config.mode + '/' + self.name_test + '_' + config.info
        if not os.path.exists(self.folder_test):
            os.makedirs(self.folder_test)
        self.folder_test = self.folder_test + '/'
 
        # print('Creating dataset...')
        self.train_dataset = SocialDataset(set_name="train", b_size=config.train_b_size, t_tresh=config.time_thresh, d_tresh=config.dist_thresh)
        self.test_dataset = SocialDataset(set_name="test", b_size=config.test_b_size, t_tresh=config.time_thresh, d_tresh=config.dist_thresh)

        if torch.cuda.is_available(): torch.cuda.set_device(config.gpu)

        self.settings = {
            "mode": config.mode,
            "use_cuda": config.cuda,
            "dim_embedding_key": config.dim_embedding_key,
            "past_len": config.past_len,
            "future_len": 1 if config.mode=='intention' else 12,
        }
        self.max_epochs = config.max_epochs

        # model
        if config.mode == 'intention':
            self.mem_n2n = model_encdec(self.settings)
        else:
            self.model_ae = torch.load(config.model_ae, map_location=torch.device('cpu')).cuda()
            self.mem_n2n = model_encdec(self.settings, self.model_ae)

        # optimizer and learning rate
        self.criterionLoss = nn.MSELoss()
        if config.mode == 'addressor':
            config.learning_rate = 1e-6
        self.opt = torch.optim.Adam(self.mem_n2n.parameters(), lr=config.learning_rate)
        if config.cuda:
            self.criterionLoss = self.criterionLoss.cuda()
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
        self.print_model_param(self.mem_n2n)
        minValue = 100
        for epoch in range(self.start_epoch, self.config.max_epochs):

            print(' ----- Epoch: {}'.format(epoch))
            loss = self._train_single_epoch()
            print('Loss: {}'.format(loss))

            if (epoch + 1) % 5 == 0:
            # if True:
                if self.config.mode == 'intention':
                    currentValue = evaluate_intention(self.test_dataset, self.mem_n2n, self.config, self.device)
                elif self.config.mode == 'addressor_warm':
                    currentValue = loss
                elif self.config.mode == 'addressor':
                    currentValue = evaluate_addressor(self.train_dataset, self.test_dataset, self.mem_n2n, self.config, self.device)
                else:
                    currentValue = evaluate_trajectory(self.test_dataset, self.mem_n2n, self.config, self.device)
                if currentValue<minValue:
                    minValue = currentValue
                    print('min value: {}'.format(minValue))
                    torch.save(self.mem_n2n, self.folder_test + 'model_ae_' + self.name_test)


    def AttentionLoss(self, sim, distance):
        dis_mask = nn.MSELoss(reduction='sum')
        threshold_distance = 80
        mask = torch.where(distance>threshold_distance, torch.zeros_like(distance), torch.ones_like(distance))
        label_sim = (threshold_distance - distance) / threshold_distance
        label_sim = torch.where(label_sim<0, torch.zeros_like(label_sim), label_sim)
        loss = dis_mask(sim*mask, label_sim*mask) / (mask.sum()+1e-5)
        return loss


    def _train_single_epoch(self):
        
        for i, (traj, mask, initial_pos,seq_start_end) \
            in enumerate(zip(self.train_dataset.trajectory_batches, self.train_dataset.mask_batches, self.train_dataset.initial_pos_batches, self.train_dataset.seq_start_end_batches)):
            traj, mask, initial_pos = torch.FloatTensor(traj).to(self.device), torch.FloatTensor(mask).to(self.device), torch.FloatTensor(initial_pos).to(self.device)
            
            initial_pose = traj[:, self.config.past_len-1, :] / 1000
            
            traj_norm = traj - traj[:, self.config.past_len-1:self.config.past_len, :]
            x = traj_norm[:, :self.config.past_len, :]
            destination = traj_norm[:, -2:, :]
            y = traj_norm[:, self.config.past_len:, :]

            abs_past = traj[:, :self.config.past_len, :]

            self.opt.zero_grad()
            if self.config.mode == 'intention':
                y = destination[:, -1:, :]
                output, recon = self.mem_n2n(x, abs_past, seq_start_end, initial_pose, destination)
                loss = self.criterionLoss(output, y) + self.criterionLoss(recon, x)
            elif self.config.mode == 'addressor_warm':
                state_past, state_past_w, memory_past, past_memory_after = self.mem_n2n(x, abs_past, seq_start_end, initial_pose, destination)
                loss = self.criterionLoss(state_past, state_past_w) + self.criterionLoss(memory_past, past_memory_after) 
            elif self.config.mode == 'addressor':
                weight_read, distance = self.mem_n2n(x, abs_past, seq_start_end, initial_pose, destination)
                loss = self.AttentionLoss(weight_read, distance)
            else:
                output, recon = self.mem_n2n(x, abs_past, seq_start_end, initial_pose, destination)
                loss = self.criterionLoss(output, y) + self.criterionLoss(recon, x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.mem_n2n.parameters(), 1.0, norm_type=2)
            self.opt.step()
        return loss.item()
