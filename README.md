# TST-ED
A temporal processes time-series model with a transformer encoder and LSTM decoder for forecasting and predicting the next event type. 

## Goals:

This model is based on temporal time processes datasets, where you are given time series sequences with each timestep containing `{t, k, v}` where `t` is the time, `k` is the event type, and `v` is the event value. In this project, we attempt to...

* predict the next time step
* predict the next event's type
* predict the next event's value

## Setup:

To run this project, you will need to install the following dependencies:
```
pytorch
numpy
wandb
easydict
pyyaml
pickle5
scipy
sklearn
matplotlib
```

## Usage:
Please ensure wandb is setup before running the model. You can run the model with the following command:
```
python Main.py -config 'configs/data_large.yaml'
```

## Acknowledgements:

A significant portion of this code is inspired by and borrowed from [Transformer Hawkes Process](https://github.com/SimiaoZuo/Transformer-Hawkes-Process) (Zou et al. 2020)