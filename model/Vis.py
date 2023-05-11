import matplotlib.pyplot as plt
import os

def visulize_processes_masks(saved_process_mask, true_process_mask, n_layers, savepath, tag):

    # if mask folder does not exist, create it
    if not os.path.exists(savepath+'/masks/'):
        os.makedirs(savepath+'/masks/')
    
    figure, axis = plt.subplots(n_layers+2, 1)
    for i in range(n_layers+1):
        axis[i].imshow(saved_process_mask[i].permute(1,0,2)[0,:,:40].cpu().detach().numpy(), cmap='Greens', interpolation='nearest', vmin=0, vmax=1)
    axis[-1].imshow(true_process_mask.permute(1,0,2)[0,:,:40].cpu().detach().numpy(), cmap='Reds', interpolation='nearest', vmin=0, vmax=1)
    plt.savefig(savepath+'/masks/process_mask_'+tag+'.png')
    plt.clf()
    plt.close('all')

def visulize_time(event_time, time_prediction, savepath, tag):

    # if testviz folder does not exist, create it
    if not os.path.exists(savepath+'/testviz/'):
        os.makedirs(savepath+'/testviz/')
    
    time_prediction_final = time_prediction.squeeze(2)

    y_zeros = [0]*len(event_time[0].cpu().detach().numpy())
    y_ones = [1]*len(event_time[0].cpu().detach().numpy())

    plt.plot(event_time[0].cpu().detach().numpy(), y_zeros, 'o', label='true')
    plt.plot(time_prediction_final[0].cpu().detach().numpy(), y_ones, 'o', label='pred')

    plt.xlim([2, 200])
    plt.ylim(-5, 5)
 
    plt.legend()
    plt.savefig(savepath+'/testviz/test_'+tag+'.png')
    plt.clf()
    plt.close('all')

