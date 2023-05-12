import torch
import model.Constants as Constants
import utils.LossFunc as LossFunc

def compute_loss(prediction, process_mask, event_time, event_type, event_process, opt, epoch):

    non_pad_mask = event_type.ne(Constants.PAD)

    """ compute SE loss """
    se, se_count = LossFunc.se_loss(prediction[0], event_time, non_pad_mask)

    """ compute bipartite matching loss """
    bp, bp_se, bp_count, type_loss, type_acc = LossFunc.bipartite_matching_loss(prediction, event_time, event_type, non_pad_mask, opt.model.num_processes)

    # SE is usually large, scale it to stabilize training
    loss = loss_manager(se, bp, type_loss, opt, epoch)

    pm_score = LossFunc.process_assignment_loss(process_mask, event_process, non_pad_mask)

    metrics = {
        'type_acc': type_acc.item(),
        'time_rmse': bp_se,
        'bp_loss': bp.item(),
        'se_loss': se.item(),
        'loss': loss.item(),
        'process_score': pm_score,
        'count_seq': event_type.size(0),
        'count_se_events': se_count,
        'count_bp_events': bp_count, 
    }

    return loss, metrics



def loss_manager(se, bp, type_loss, opt, epoch):

    scale_time_loss = 100
    loss = 0
    
    if 'se' in opt.model.loss_mode:
        loss += se
    
    if 'bp' in opt.model.loss_mode:
        factor = min(epoch / opt.train.bp_start_epoch, 0.8)
        loss += factor*bp + (1-factor)*se

    loss = loss / scale_time_loss

    if 'type' in opt.model.loss_mode:
        loss += type_loss

    return loss

