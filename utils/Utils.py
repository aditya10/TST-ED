import torch
import os
import wandb

def setup_process_mask(event_process, opt):

    # 1: Build GT process mask [P, B, S] from event_process [B, S]
    process_mask_gt = event_process.unsqueeze(0).repeat(opt.model.num_processes, 1, 1).to(opt.device)
    for i in range(opt.model.num_processes):
        process_mask_gt[i] = (process_mask_gt[i] == i+1).long()

    # 2: Build inital process mask [P, B, S] where any element can be from any process
    process_mask = (1/opt.model.num_processes)*torch.ones_like(process_mask_gt).to(opt.device)

    return process_mask_gt, process_mask


def save_checkpoint(model, optimizer, epoch, checkpoint_dir):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'wandb_id': wandb.run.id,
            }, os.path.join(checkpoint_dir, 'checkpoint_{}.pt'.format(epoch)))


def load_checkpoint(model, checkpoint_dir, epoch, optimizer=None):

    checkpoint = torch.load(os.path.join(checkpoint_dir, 'checkpoint_{}.pt'.format(epoch)))
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['wandb_id']