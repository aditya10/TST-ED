import math
import torch
import torch.nn as nn

import model.Constants as Constants
from model.Layers import MPEncoderLayer


def get_attn_key_pad_mask(seq_k, seq_q):
    """ For masking out the padding part of key sequence. """

    # expand to fit the shape of key query attention matrix
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask


def get_subsequent_mask(seq):
    """ For masking out the subsequent info, i.e., masked self-attention. """

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
    return subsequent_mask


class MPEncoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self,
            num_types, d_model, d_inner,
            n_layers, n_head, d_k, d_v, dropout, 
            num_processes, oracle, viz=False):
        super().__init__()

        self.d_model = d_model
        self.num_processes = num_processes
        self.n_layers = n_layers
        self.viz = viz

        # position vector, used for temporal encoding
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)],
            device=torch.device('cuda'))

        # event type embedding
        self.event_emb = nn.Embedding(num_types + 1, d_model, padding_idx=Constants.PAD)
        self.value_emb = nn.Linear(1, d_model)
        self.time_value_combo = nn.Linear(d_model * 2, d_model)

        self.layer_stack = nn.ModuleList([
            MPEncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, 
            normalize_before=False, num_processes=num_processes, oracle=oracle)
            for _ in range(n_layers)])

    def temporal_enc(self, time, non_pad_mask):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """

        result = time.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result * non_pad_mask

    def forward(self, event_type, event_time, non_pad_mask, process_mask, event_value=None):
        """ Encode event sequences via masked self-attention. """

        # prepare attention masks
        # slf_attn_mask is where we cannot look, i.e., the future and the padding
        slf_attn_mask_subseq = get_subsequent_mask(event_type)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=event_type, seq_q=event_type)
        slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        tem_enc = self.temporal_enc(event_time, non_pad_mask)
        enc_output = self.event_emb(event_type)
        if event_value is not None:
            val_enc = self.value_emb(event_value.unsqueeze(-1)) 
            enc_output = self.time_value_combo(torch.cat([enc_output, val_enc], dim=-1))

        saved_process_mask = []
        for enc_layer in self.layer_stack:
            enc_output += tem_enc
            saved_process_mask.append(process_mask)
            enc_output, _, process_mask = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                process_mask=process_mask)

        saved_process_mask.append(process_mask)

        return enc_output, saved_process_mask