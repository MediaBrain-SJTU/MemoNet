import torch



def evaluate_intention(dataset, model, config, device): 
    fde_48s = 0
    samples = 0

    with torch.no_grad():
        for i, (traj, mask, initial_pos,seq_start_end) \
            in enumerate(zip(dataset.trajectory_batches, dataset.mask_batches, dataset.initial_pos_batches, dataset.seq_start_end_batches)):
            traj, mask, initial_pos = torch.FloatTensor(traj).to(device), torch.FloatTensor(mask).to(device), torch.FloatTensor(initial_pos).to(device)
            # traj (B, T, 2)
            initial_pose = traj[:, config.past_len-1, :] / 1000
            
            traj_norm = traj - traj[:, config.past_len-1:config.past_len, :]
            x = traj_norm[:, :config.past_len, :]
            destination = traj_norm[:, -2:, :]
            y = traj_norm[:, -1:, :]

            abs_past = traj[:, :config.past_len, :]

            output, _ = model(x, abs_past, seq_start_end, initial_pose, destination)
            output = output.data

            distances = torch.norm(output - y, dim=2)
            fde_48s += torch.sum(distances[:, -1])
            samples += distances.shape[0]

    return fde_48s / samples



def evaluate_addressor(train_dataset, dataset, model, config, device): 
    fde_48s = 0
    samples = 0

    model.generate_memory(train_dataset, filter_memory=False)

    with torch.no_grad():
        for i, (traj, mask, initial_pos,seq_start_end) \
            in enumerate(zip(dataset.trajectory_batches, dataset.mask_batches, dataset.initial_pos_batches, dataset.seq_start_end_batches)):
            traj, mask, initial_pos = torch.FloatTensor(traj).to(device), torch.FloatTensor(mask).to(device), torch.FloatTensor(initial_pos).to(device)
            # traj (B, T, 2)
            initial_pose = traj[:, config.past_len-1, :] / 1000
            
            traj_norm = traj - traj[:, config.past_len-1:config.past_len, :]
            x = traj_norm[:, :config.past_len, :]
            y = traj_norm[:, -1:, :]

            abs_past = traj[:, :config.past_len, :]

            output = model.get_destination_from_memory(x, abs_past, seq_start_end, initial_pose)
            output = output.data

            future_rep = y.unsqueeze(1).repeat(1, 20, 1, 1)
            distances = torch.norm(output - future_rep, dim=3)
            mean_distances = torch.mean(distances[:, :, -1:], dim=2)
            index_min = torch.argmin(mean_distances, dim=1)
            min_distances = distances[torch.arange(0, len(index_min)), index_min]

            fde_48s += torch.sum(min_distances[:, -1])
            samples += distances.shape[0]

    return fde_48s / samples


def evaluate_trajectory(dataset, model, config, device):
    # for the fulfillment stage or trajectory stage, we should have a fixed past/intention memory bank.
    ade_48s = fde_48s = 0
    samples = 0
    dict_metrics = {}

    with torch.no_grad():
        for i, (traj, mask, initial_pos,seq_start_end) \
            in enumerate(zip(dataset.trajectory_batches, dataset.mask_batches, dataset.initial_pos_batches, dataset.seq_start_end_batches)):
            traj, mask, initial_pos = torch.FloatTensor(traj).to(device), torch.FloatTensor(mask).to(device), torch.FloatTensor(initial_pos).to(device)
            # traj (B, T, 2)
            initial_pose = traj[:, config.past_len-1, :] / 1000
            
            traj_norm = traj - traj[:, config.past_len-1:config.past_len, :]
            x = traj_norm[:, :config.past_len, :]
            destination = traj_norm[:, -2:, :]

            abs_past = traj[:, :config.past_len, :]

            output = model.get_trajectory(x, abs_past, seq_start_end, initial_pose)
            output = output.data

            future_rep = traj_norm[:, 8:, :].unsqueeze(1).repeat(1, 20, 1, 1)
            distances = torch.norm(output - future_rep, dim=3)
            mean_distances = torch.mean(distances[:, :, -1:], dim=2) # find the tarjectory according to the last frame's distance
            index_min = torch.argmin(mean_distances, dim=1)
            min_distances = distances[torch.arange(0, len(index_min)), index_min]

            fde_48s += torch.sum(min_distances[:, -1])
            ade_48s += torch.sum(torch.mean(min_distances, dim=1))
            samples += distances.shape[0]


        dict_metrics['fde_48s'] = fde_48s / samples
        dict_metrics['ade_48s'] = ade_48s / samples

    return dict_metrics['ade_48s']
