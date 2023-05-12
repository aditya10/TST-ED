import torch
import torch.nn as nn

import model.Constants as Constants
from model.Encoder import MPEncoder
from model.Predictors import Predictor, NumberPredictor


def get_non_pad_mask(seq):
    """ Get the non-padding positions. """

    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)


class MPLSTM(nn.Module):
    """
    Optional recurrent layers. This is inspired by the fact that adding
    recurrent layers on top of the Transformer helps language modeling.
    """

    def __init__(self, d_model, d_rnn, num_processes):
        super().__init__()

        self.process_projection = nn.Linear(num_processes, d_model)
        self.rnn = nn.LSTM(d_model*2, d_rnn, num_layers=1, batch_first=True)
        self.projection = nn.Linear(d_rnn, d_model)

    def forward(self, enc_data, process_mask, non_pad_mask):
        # enc_data: [B,S,D]
        # process_mask: [P,B,S]

        process_data = self.process_projection(process_mask.permute(1,2,0)) # [B,S,P] -> [B,S,D]
        data = torch.cat([enc_data, process_data], dim=-1)

        lengths = non_pad_mask.squeeze(2).long().sum(1).cpu()
        pack_enc_output = nn.utils.rnn.pack_padded_sequence(
            data, lengths, batch_first=True, enforce_sorted=False)
        temp = self.rnn(pack_enc_output)[0]
        out = nn.utils.rnn.pad_packed_sequence(temp, batch_first=True)[0]

        out = self.projection(out)
        return out

class MPTransformer(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(
            self,
            num_types, d_model=256, d_rnn=128, d_inner=1024,
            n_layers=4, n_head=4, d_k=64, d_v=64, dropout=0.1,
            num_processes=1, p_init_random=False, oracle=False, viz=False):
        super().__init__()

        self.num_types = num_types
        self.num_processes = num_processes
        self.p_init_random = p_init_random
        self.viz = viz

        self.encoder = MPEncoder(
            num_types=num_types,
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            num_processes=num_processes,
            oracle=oracle,
            viz=self.viz,
        )

        # OPTIONAL recurrent layer, this sometimes helps
        self.lstm = MPLSTM(d_model, d_rnn, num_processes)

        self.time_predictor = NumberPredictor(d_model)

        # prediction of next event type
        self.type_predictor = Predictor(d_model, num_types)

        # prediction of next event value
        self.value_predictor = NumberPredictor(d_model)

    def forward(self, event_time, event_type, event_value, process_mask):
        """
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input: event_type: batch*seq_len;
               event_time: batch*seq_len;
               event_value: batch*seq_len.
        Output: enc_output: batch*seq_len*model_dim;
                type_prediction: batch*seq_len*num_classes (not normalized);
                time_prediction: batch*seq_len;
                value_prediction: batch*seq_len.
        """

        non_pad_mask = get_non_pad_mask(event_type)
        # Encoder
        enc_output, saved_process_mask = self.encoder(event_type, event_time, non_pad_mask, process_mask, event_value)
        # Decoder, conditioned on processes
        enc_output = self.lstm(enc_output, saved_process_mask[-1], non_pad_mask)

        # Predictor
        time_diff_prediction = self.time_predictor(enc_output, non_pad_mask)
        time_prediction = event_time + time_diff_prediction.squeeze(-1)
        time_prediction.unsqueeze_(-1)
        type_prediction = self.type_predictor(enc_output, non_pad_mask)
        value_prediction = self.value_predictor(enc_output, non_pad_mask)

        return enc_output, (time_prediction, type_prediction, value_prediction), saved_process_mask
    
