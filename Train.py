import numpy as np
import time
import torch

import utils.Utils as Utils
import model.Vis as Vis
import utils.Loss as Loss
from Eval import eval_epoch

from tqdm import tqdm
import wandb


def train_epoch(model, training_data, optimizer, opt, epoch):
    """ Epoch operation in training phase. """

    model.train()

    start_time = time.time()

    metrics = {
        'type_acc': 0, # cumulative accuracy of type prediction
        'time_rmse': 0, # cumulative time prediction squared-error
        'bp_loss': 0, # cumulative bipartitle loss
        'se_loss': 0, # cumulative time prediction squared-error
        'loss': 0, # cumulative loss
        'process_score': 0, # cumulative process score
        'count_seq': 0, # cumulative number of sequences
        'count_se_events': 0, # cumulative number of events for squared-error
        'count_bp_events': 0, # cumulative number of events for bipartitle loss
    }

    viz_done = False

    for batch in tqdm(training_data, mininterval=2, desc=' - (Training) ', leave=False):


        """ prepare data """
        event_time, event_type, event_process = map(lambda x: x.to(opt.device), batch)
        event_value = None

        """ setup process mask """
        process_mask_gt, process_mask = Utils.setup_process_mask(event_process, opt)

        """ forward """
        optimizer.zero_grad()
        
        enc_out, prediction, saved_process_masks = model(event_time, event_type, event_value, process_mask)
        
        """ visualize """
        if opt.visualize and epoch % 100 == 0 and not viz_done:
            Vis.visulize_processes_masks(saved_process_masks, process_mask_gt, opt.model.n_layers, opt.path, 'train'+str(epoch))
            Vis.visulize_time(event_time, prediction[0], opt.path, 'train'+str(epoch))
            viz_done = True

        """ backward """
        loss, batch_metrics = Loss.compute_loss(prediction, saved_process_masks[-1], event_time, event_type, event_process, opt, epoch)
        
        loss.backward()

        """ update parameters """
        optimizer.step()

        """ update cumulative metrics """
        for k, v in batch_metrics.items():
            metrics[k] += round(float(v), 3)

    """ normalize cumulative metrics """
    metrics['type_acc'] = round(metrics['type_acc'] / metrics['count_bp_events'], 3)
    metrics['time_rmse'] = round(np.sqrt(metrics['time_rmse'] / metrics['count_bp_events']), 3)
    metrics['process_score'] = round(metrics['process_score'] / metrics['count_seq'], 3)
    
    metrics['time_per_epoch'] = round(time.time() - start_time, 3)

    return {'train': metrics}


def train(model, training_data, validation_data, optimizer, scheduler, opt, checkpoint_after=500):
    """ Start training. """

    for epoch_i in range(opt.train.epoch):
        epoch = epoch_i + 1
        print('[ Epoch', epoch, ']')

        # Train epoch
        train_metrics = train_epoch(model, training_data, optimizer, opt, epoch)

        print('  - (Training) ', train_metrics)
        wandb.log(train_metrics, step=epoch)

        # Validation epoch
        val_metrics = eval_epoch(model, validation_data, opt, epoch)
        print('  - (Validation) ', val_metrics)
        wandb.log(val_metrics, step=epoch)
        
        if epoch % checkpoint_after == 0:
            Utils.save_checkpoint(model, optimizer, epoch, opt.path)
    
        # logging
        with open(opt.log, 'a') as f:
            f.write('Train: '+str(train_metrics))
            f.write('Val: '+str(val_metrics))

        # update learning rate
        scheduler.step()