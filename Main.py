import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
#import pickle
from pickle5 import pickle
import datetime
import os
import yaml
from yaml import CLoader as Loader
from easydict import EasyDict as edict
import random
import wandb

from Train import train
import utils.Utils as Utils
from preprocess.Dataset import get_dataloader
from model.Models import MPTransformer

torch.set_printoptions(sci_mode=False)

def prepare_dataloader(opt):
    """ Load data and prepare dataloader. """

    def load_data(name, dict_name):
        with open(name, 'rb') as f:
            data = pickle.load(f, encoding='latin-1')
            num_types = data['dim_process']
            data = data[dict_name]
            return data, int(num_types)

    print('[Info] Loading train data...')
    train_data, num_types = load_data(opt.data.path + 'train.pkl', 'train')
    print('[Info] Loading dev data...')
    dev_data, _ = load_data(opt.data.path + 'dev.pkl', 'dev')
    print('[Info] Loading test data...')
    test_data, _ = load_data(opt.data.path + 'test.pkl', 'test')

    trainloader = get_dataloader(train_data, opt, shuffle=True)
    testloader = get_dataloader(test_data, opt, shuffle=False)
    
    return trainloader, testloader, num_types


def main():
    """ Main function. """

    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, default='config.json', help='Path to the config file.')
    args = parser.parse_args()

    """ load config """

    with open(args.config) as f:
        opt = yaml.load(f, Loader)
        opt = edict(opt)

    wandb_name = opt.data.path+'/'+opt.data.mode[0]+'-'+str(opt.model.num_processes)+datetime.datetime.now().strftime("/%H%M%S-%B%d")
    wandb.init(project='TST-ED', name=wandb_name, config=opt, mode=("online" if opt.use_wandb else "disabled"))

    # default device is CUDA
    opt.device = torch.device('cuda')

    # create a folder for saving checkpoints
    opt.path = opt.checkpoint_dir + wandb_name
    if not os.path.exists(opt.path):
        os.makedirs(opt.path)

    # save the config file
    with open(opt.path + '/config.yaml', 'w') as f:
        yaml.dump(opt, f)

    # setup the log file
    opt.log = opt.path + '/log.txt'
    with open(opt.log, 'w') as f:
        f.write('Epoch, Accuracy, RMSE\n')

    print('[Info] parameters: \n', opt)

    """ prepare dataloader """
    trainloader, testloader, num_types = prepare_dataloader(opt)

    """ set seeds """
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)

    """ prepare model """
    model = MPTransformer(
        num_types=num_types,
        d_model=opt.model.d_model,
        d_rnn=opt.model.d_rnn,
        d_inner=opt.model.d_inner,
        n_layers=opt.model.n_layers,
        n_head=opt.model.n_head,
        d_k=opt.model.d_k,
        d_v=opt.model.d_v,
        dropout=opt.model.dropout,
        num_processes=opt.model.num_processes,
        p_init_random=opt.model.p_init_random,
        oracle=opt.model.oracle,
        viz=opt.visualize
    )
    model.to(opt.device)

    wandb.watch(model)

    """ optimizer and scheduler """
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                           opt.train.lr, betas=(0.9, 0.999), eps=1e-05)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [10, 50, 200], gamma=0.5)

    """ number of parameters """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('[Info] Number of parameters: {}'.format(num_params))

    """ train the model """
    train(model, trainloader, testloader, optimizer, scheduler, opt)


if __name__ == '__main__':
    main()
