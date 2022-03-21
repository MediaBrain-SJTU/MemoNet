import os
import datetime
import torch
import torch.nn as nn

from models.model_selector_warm_up_social import model_encdec


from sddloader import *
torch.set_num_threads(5)

class Trainer:
    def __init__(self, config):
        
        self.name_test = str(datetime.datetime.now())[:10]
        self.folder_test = 'training/training_selector/' + self.name_test + '_' + config.info
        if not os.path.exists(self.folder_test):
            os.makedirs(self.folder_test)
        self.folder_test = self.folder_test + '/'
 
        
        torch.set_default_dtype(torch.float32)
        device = torch.device('cuda', index=config.gpu) if torch.cuda.is_available() else torch.device('cpu')
        if torch.cuda.is_available(): torch.cuda.set_device(config.gpu)
        
        
        self.train_dataset = SocialDataset(set_name="train", b_size=config.train_b_size, t_tresh=config.time_thresh, d_tresh=config.dist_thresh)

        self.test_dataset = SocialDataset(set_name="test", b_size=config.test_b_size, t_tresh=config.time_thresh, d_tresh=config.dist_thresh)


        self.settings = {
            "use_cuda": config.cuda,
            "dim_feature_tracklet": config.past_len * 2,
            "dim_feature_future": config.future_len * 2,
            "dim_embedding_key": config.dim_embedding_key,
            "past_len": config.past_len,
            "future_len": 1,
        }
        self.max_epochs = config.max_epochs

        # model
        self.model_ae = torch.load(config.model_ae, map_location=device)
        self.mem_n2n = model_encdec(self.settings, self.model_ae)

        # loss
        self.criterionLoss = nn.MSELoss()

        self.opt = torch.optim.Adam(self.mem_n2n.parameters(), lr=config.learning_rate)
        self.iterations = 0
        if config.cuda:
            self.criterionLoss = self.criterionLoss.cuda()
            self.mem_n2n = self.mem_n2n.cuda()
        
        self.config = config


    def fit(self):
        
        config = self.config
        # Training loop
        for epoch in range(0, config.max_epochs):

            print(' ----- Epoch: {}'.format(epoch))
            loss = self._train_single_epoch()
            print('Loss: {}'.format(loss))

            torch.save(self.mem_n2n, self.folder_test + 'model_ae_epoch_' + str(epoch) + '_' + self.name_test)

      

    def _train_single_epoch(self):

        for i, (traj, mask, initial_pos,seq_start_end) \
            in enumerate(zip(self.train_dataset.trajectory_batches, self.train_dataset.mask_batches, self.train_dataset.initial_pos_batches, self.train_dataset.seq_start_end_batches)):
            traj, mask, initial_pos = torch.FloatTensor(traj).cuda(), torch.FloatTensor(mask).cuda(), torch.FloatTensor(initial_pos).cuda()
           
            initial_pose = traj[:, 7, :] / 1000
            
            traj_norm = traj - traj[:, 7:8, :]
            x = traj_norm[:, :self.config.past_len, :]
            destination = traj_norm[:, -2:, :]
            y = destination[:, -1:, :]

            abs_past = traj[:, :self.config.past_len, :]

            self.opt.zero_grad()
            state_past, state_past_w, memory_past, past_memory_after = self.mem_n2n(x, abs_past, seq_start_end, initial_pose, destination)


            loss = self.criterionLoss(state_past, state_past_w) + self.criterionLoss(memory_past, past_memory_after)
            loss.backward()
            self.opt.step()


        return loss.item()
