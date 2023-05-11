import torch
import torch.nn as nn


class Predictor(nn.Module):
    """ Prediction of next event type. """

    def __init__(self, dim, num_types, leakyRelu=False):
        super().__init__()

        self.leakyRelu = leakyRelu

        self.linear1 = nn.Linear(dim, dim*2, bias=True)
        self.activation1 = nn.ReLU()
        self.linear2 = nn.Linear(dim*2, dim*2, bias=True)
        self.activation2 = nn.ReLU()
        self.linear3 = nn.Linear(dim*2, num_types, bias=True)
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)
        nn.init.xavier_normal_(self.linear3.weight)
        if self.leakyRelu:
            self.activation3 = nn.LeakyReLU()

    def forward(self, data, non_pad_mask):
        out = self.linear1(data)
        out = self.activation1(out)
        out = self.linear2(out)
        out = self.activation2(out)
        out = self.linear3(out)

        if self.leakyRelu:
            out = self.activation3(out)

        out = out * non_pad_mask
        return out

class NumberPredictor(nn.Module):
    ''' Prediction of next time stamp. '''

    def __init__(self, dim):
        super().__init__()

        self.linear1 = nn.Linear(dim, dim*2, bias=True)
        self.activation1 = nn.ReLU()
        self.linear2 = nn.Linear(dim*2, 1, bias=True)

        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)

    def forward(self, data, non_pad_mask):
        out = self.linear1(data)
        out = self.activation1(out)
        out = self.linear2(out)
        out = torch.exp(out)
        out = out * non_pad_mask
        return out