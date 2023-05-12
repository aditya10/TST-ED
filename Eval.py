import numpy as np
import time
import torch

import model.Vis as Vis
import utils.Loss as Loss
import utils.Utils as Utils

from tqdm import tqdm


def eval_epoch(model, validation_data, opt, epoch):

    """ Epoch operation in evaluation phase. """

    model.eval()

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

    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2, desc=' - (Validation) ', leave=False):

            """ prepare data """
            event_time, event_type, event_process = map(lambda x: x.to(opt.device), batch)
            event_value = None

            """ setup process mask """
            process_mask_gt, process_mask = Utils.setup_process_mask(event_process, opt)

            """ forward """
            enc_out, prediction, saved_process_masks = model(event_time, event_type, event_value, process_mask)

            """ visualize """
            if opt.visualize and epoch % 100 == 0 and not viz_done:
                Vis.visulize_processes_masks(saved_process_masks, process_mask_gt, opt.model.n_layers, opt.path, 'val'+str(epoch))
                Vis.visulize_time(event_time, prediction[0], opt.path, 'val'+str(epoch))
                viz_done = True

            """ compute loss """
            loss, batch_metrics = Loss.compute_loss(prediction, saved_process_masks[-1], event_time, event_type, event_process, opt, epoch)

            """ update cumulative metrics """
            for k, v in batch_metrics.items():
                metrics[k] += round(float(v), 3)

    """ normalize cumulative metrics """
    metrics['type_acc'] = round(metrics['type_acc'] / metrics['count_bp_events'], 3)
    metrics['time_rmse'] = round(np.sqrt(metrics['time_rmse'] / metrics['count_bp_events']), 3)
    metrics['process_score'] = round(metrics['process_score'] / metrics['count_seq'], 3)
    
    metrics['time_per_epoch'] = round(time.time() - start_time, 3)

    return {'val': metrics}