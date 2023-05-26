import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
import torch.nn.functional as F


def se_loss(prediction, event_time, non_pad_mask):
    """ Time prediction loss. """

    prediction.squeeze_(-1)

    se_total = 0
    for i in range(prediction.size(0)):
        pred = prediction[i, non_pad_mask[i]]
        true = event_time[i, non_pad_mask[i]]

        true = true[1:] 
        pred = pred[:-1]
        
        diff = pred - true
        se = torch.sum(diff * diff)

        se_total += se
    
    se_count = torch.sum(non_pad_mask).item() - prediction.size(0)

    return se_total, se_count


def bipartite_matching_loss(prediction, event_time, event_type, non_pad_mask, num_processes):

    def bipartite_matching_loss_algo(pred, true, num_processes):
    
        costs = torch.zeros((pred.shape[0]+num_processes-1, true.shape[0]+num_processes-1))
        costs[:pred.shape[0], :true.shape[0]] = (pred.unsqueeze(1) - true.unsqueeze(0))**2
        costs.fill_diagonal_(10000)
        costs[pred.shape[0]:, true.shape[0]:] = 0

        row_ind, col_ind = linear_sum_assignment(costs.detach().cpu().numpy())

        count_matches = pred.shape[0]
        good_row_idx = []
        good_col_idx = []
        for i in range(len(row_ind)):
            if row_ind[i] > pred.shape[0]-1 and col_ind[i] <= true.shape[0]-1:
                count_matches -= 1
            elif row_ind[i] <= pred.shape[0]-1 and col_ind[i] <= true.shape[0]-1:
                good_row_idx.append(row_ind[i])
                good_col_idx.append(col_ind[i])
        
        return costs[row_ind, col_ind].sum(), count_matches, (good_row_idx, good_col_idx)

    time_prediction, type_prediction, value_prediction = prediction
    
    time_prediction.squeeze_(-1)

    loss_total = 0
    se_total = 0
    count_total = 0
    type_loss_total = 0
    type_acc_total = 0
    for i in range(time_prediction.size(0)):
        pred = time_prediction[i, non_pad_mask[i]]
        true = event_time[i, non_pad_mask[i]]
    
        loss, count, good_matches = bipartite_matching_loss_algo(pred, true, num_processes)

        loss_total += loss
        se_total += loss.item()
        count_total += count

        type_true = event_type[i, non_pad_mask[i]]
        type_pred = type_prediction[i, non_pad_mask[i]]

        type_loss, type_acc = type_loss_algo(type_pred, type_true, good_matches)

        type_loss_total += type_loss
        type_acc_total += type_acc

    return loss_total, se_total, count_total, type_loss_total, type_acc_total


# Loss for process assignment, use sklearn's adjusted_rand_score
def process_assignment_loss(saved_process_masks, event_process, non_pad_mask):

    process_mask = saved_process_masks[1].detach().clone()
    for m in saved_process_masks[2:]:
        process_mask += m.detach().clone()
    
    process_mask = process_mask.permute(1, 2, 0) # [B, S, P]

    # To test random process assignment
    # process_mask = torch.rand_like(process_mask)
    
    # # To test constant assignment
    # process_mask[:, :, 0] = 1
    # process_mask[:, :, 1:] = 0
    
    # pick process with highest probability
    process_mask = torch.argmax(process_mask, dim=2) + 1 # [B, S]

    total_ars = 0
    for i in range(process_mask.shape[0]):
        pred = process_mask[i][non_pad_mask[i]]
        true = event_process[i][non_pad_mask[i]]

        total_ars += adjusted_rand_score(pred.cpu().detach().numpy(), true.cpu().detach().numpy())

    return total_ars

def type_loss_algo(pred, true, good_matches):

    # pred: [S, num_types]
    # true: [S]

    (good_row_idx, good_col_idx) = good_matches

    pred = pred[good_row_idx]
    true = true[good_col_idx] - 1 # -1 because event_type starts from 1

    # use cross entropy loss
    loss = F.cross_entropy(pred, true, reduction='sum')

    # compute accuracy
    pred_type = torch.max(pred, dim=-1)[1]
    correct_num = torch.sum(pred_type == true)

    return loss, correct_num
