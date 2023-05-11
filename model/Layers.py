import torch.nn as nn
import torch

from model.Modules import MPMultiHeadAttention, PositionwiseFeedForwardNoNorm, ProcessMaskPredictor


class MPEncoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, normalize_before=True, num_processes=2, oracle=False):
        super(MPEncoderLayer, self).__init__()

        self.num_processes = num_processes
        self.oracle = oracle

        self.slf_attn = MPMultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout, normalize_before=normalize_before)

        self.pos_ffn = nn.ModuleList([PositionwiseFeedForwardNoNorm(
            d_model, d_inner, dropout=dropout) for _ in range(num_processes)])

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.process_mask_predictor = ProcessMaskPredictor(d_model, num_processes)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None, process_mask=None):

        # Input shapes: enc_input [B,S,D], non_pad_mask [B,S,1] 0 when mask, slf_attn_mask [B,S,S] True when mask, else false, process_mask [P,B,S] 0 when mask

        # Step 1: Feed input [B,S,D] and mask [P,B,1,S,S] into MultiProcessMHA. Output [P,B,S,D]
        # Step 2: Feed each of the P outputs [B,S,D] into a separate MultiProcessPositionwiseFeedForward. Output is a list of [B,S,D] of length P
        # Step 3: Mask the outputs using P masks [B,S]. Sum the P outputs across the S dim to get [B,S,D] and normalize. Output [B,S,D]

        # Step 1: Generate the final mask for attention, feed to MHA.
        slf_attn_mask_l = (~slf_attn_mask).long()
        final_masks = []
        for i in range(self.num_processes):
            process_mask_attn = process_mask[i].unsqueeze(-1) # (B, S, 1)
            process_mask_attn = torch.bmm(process_mask_attn.float(), process_mask_attn.float().transpose(2,1))
            process_mask_attn = slf_attn_mask_l.float() * process_mask_attn.float()
            final_masks.append(process_mask_attn)

        final_mask = torch.stack(final_masks, dim=0) # (P, B, S, S)
        final_mask = final_mask.unsqueeze(2)

        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=final_mask)

        # Step 2: feed to individal positionwise feedforward layers
        encoded_outputs = []
        for i in range(self.num_processes):
            enc_output_i = enc_output[i] # (B, N, D)
            enc_output_i = self.pos_ffn[i](enc_output_i)
            enc_output_i *= process_mask[i].unsqueeze(-1)
            encoded_outputs.append(enc_output_i)

        # Step 3: merge all the encoded outputs, normalize
        enc_output = torch.stack(encoded_outputs, dim=0) # (P, B, N, D)
        enc_output = enc_output.sum(dim=0) # (B, N, D)
        enc_output = self.layer_norm(enc_output)

        enc_output *= non_pad_mask

        enc_output_in = enc_output.clone()

        # predict process type, of shape (T, B, N)
        if not self.oracle:
            process_mask = self.process_mask_predictor(enc_output_in, process_mask)

        return enc_output, enc_slf_attn, process_mask