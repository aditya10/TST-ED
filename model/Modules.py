import torch
import torch.nn as nn
import torch.nn.functional as F


class MPScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention with K masks leading to an output size of [B,K,V]"""

    def __init__(self, temperature, attn_dropout=0.2):
        super().__init__()

        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        # Inputs: q, k, v are of shape [B, H, S, D], mask [P, B, 1, S, S]

        # Step 1: Compute attention scores [B, H, S, S]
        # Step 2: Mask the attention scores using the mask [P, B, 1, S, S] for each of the P masks
        # Step 3: Compute the softmax of the masked attention scores [B, H, S, S]
        # Step 4: Stack the P outputs [P, B, H, S, D]

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        # Mask where 1 is where to attend and 0 is where to ignore. 
        # https://github.com/huggingface/transformers/issues/542
        extended_attention_mask = (1.0 - mask) * -10000.0 

        outputs = []
        for i in range(mask.shape[0]):

            attn_masked = attn + extended_attention_mask[i]

            attn_masked = self.dropout(F.softmax(attn_masked, dim=-1))
            output_p = torch.matmul(attn_masked, v)
            outputs.append(output_p)

        output = torch.stack(outputs)

        return output, attn


class MPMultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, normalize_before=True):
        super().__init__()

        self.normalize_before = normalize_before
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.xavier_uniform_(self.w_qs.weight)
        nn.init.xavier_uniform_(self.w_ks.weight)
        nn.init.xavier_uniform_(self.w_vs.weight)

        self.fc = nn.Linear(d_v * n_head, d_model)
        nn.init.xavier_uniform_(self.fc.weight)

        self.attention = MPScaledDotProductAttention(temperature=d_k ** 0.5, attn_dropout=dropout)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):

        # Inputs shapes: q [B, S, D], k [B, S, D], v [B, S, D], mask [P, B, 1, S, S]

        # Step 1: Obtain new k, q, v using the linear layers for each head, [B, S, H, D]
        # Step 2: Transpose the dimensions to [B, H, S, D]
        # Step 3: Apply the attention function with input [B, H, S, D], and mask [P, B, 1, S, S]. Output [P, B, H, S, D]
        # Step 4: Transpose the dimensions to [P, B, S, H, D]
        # Step 5: Concatenate the heads to [P, B, S, D*H]
        # Step 6: Apply the linear layer. Output [P, B, S, D]
        # Step 7: Expand the residual to [P, B, S, D], then add and normalize per process P. Output [P, B, S, D]

        # Step 1
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        num_p = mask.size(0)

        residual = q
        if self.normalize_before:
            q = self.layer_norm(q)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Step 2
        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # Step 3
        output, attn = self.attention(q, k, v, mask=mask)
        # the output here is of the following shape: [P, B, H, S, D]

        # Step 4,5,6
        # Transpose to move the head dimension back: P x B x S x H x dv
        # Combine the last two dimensions to concatenate all the heads together: P x B x S x (H*dv)
        output = output.transpose(2, 3).contiguous().view(num_p, sz_b, len_q, -1)
        output = self.dropout(self.fc(output))

        # Step 7
        output += residual.unsqueeze(0).repeat(num_p, 1, 1, 1)

        if not self.normalize_before:
            output = self.layer_norm(output)
        return output, attn


class PositionwiseFeedForwardNoNorm(nn.Module):
    """ Two-layer position-wise feed-forward neural network. """

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()

        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.w_1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        x = x + residual
        return x
    

class ProcessMaskPredictor(nn.Module):
    """ Prediction of next event type. """

    def __init__(self, d_model, num_processes):
        super().__init__()
        self.process_predictor_linear = nn.Linear(d_model, num_processes)
        self.process_predictor_softmax = nn.Softmax(dim=-1)

    def forward(self, enc_output_in, process_mask):

        process_type_logits = self.process_predictor_linear(enc_output_in)
        process_type_predictions = self.process_predictor_softmax(process_type_logits)
        process_mask_with_residual = torch.softmax(process_type_predictions, dim=-1).permute(2,0,1)+process_mask
        process_mask=softmax(process_mask_with_residual, dim=0, t=0.5)
        
        return process_mask

  
def softmax(input, dim=0, t=1.0):
  ex = torch.exp(input/t)
  sum = torch.sum(ex, axis=dim)
  return ex / sum