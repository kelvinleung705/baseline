import copy
import math
import torch
from torch import nn
import torch.nn.functional as F

from models.LayerNormGRU import LayerNormGRU
from transformers import BertConfig, BertForMaskedLM

batch_first = False
bidirectional = False


class MulT_TTE(nn.Module):
    def __init__(self, input_dim, seq_input_dim, seq_hidden_dim, seq_layer, bert_hiden_size, pad_token_id,
                 bert_attention_heads, bert_hidden_layers, decoder_layer, decode_head, vocab_size=27300):
        super(MulT_TTE, self).__init__()
        self.bert_config = BertConfig(
            num_attention_heads=bert_attention_heads,
            hidden_size=bert_hiden_size,
            pad_token_id=pad_token_id,
            vocab_size=vocab_size,
            num_hidden_layers=bert_hidden_layers
        )
        self.seg_embedding_learning = BertForMaskedLM(self.bert_config)

        self.highwayembed = nn.Embedding(15, 5, padding_idx=0)
        self.gpsrep = nn.Linear(4, 16)

        # Process 14-dim pre-encoded global context:
        # Time(2) + DayOfWeek(7) + Month(2) + Holiday(1) + Visibility(1) + SnowDepth(1) = 14 dims
        self.context_dim = 32
        self.context_mlp = nn.Sequential(
            nn.Linear(14, self.context_dim),
            nn.LeakyReLU(),
            nn.Linear(self.context_dim, self.context_dim)
        )

        self.timene_dim = self.context_dim + bert_hiden_size

        self.timene = nn.Sequential(
            nn.Linear(self.timene_dim, self.timene_dim),
            nn.LeakyReLU(),
            nn.Linear(self.timene_dim, self.timene_dim)
        )

        # Input to represent: seg_len & cum_length (2) + highwayrep (5) + gpsrep (16) + timene (timene_dim) + dynamic_feats (4)
        represent_input_dim = 2 + 5 + 16 + self.timene_dim + 4

        self.represent = nn.Sequential(
            nn.Linear(represent_input_dim, seq_input_dim),
            nn.LeakyReLU(),
            nn.Linear(seq_input_dim, seq_input_dim)
        )

        self.sequence = LayerNormGRU(seq_input_dim, seq_hidden_dim, seq_layer)
        self.seq_hidden_dim = seq_hidden_dim * 2 if bidirectional else seq_hidden_dim
        self.decoder_embed_dim = seq_hidden_dim * 2 if bidirectional else seq_hidden_dim

        self.input2hid = nn.Linear(seq_hidden_dim + self.context_dim, seq_hidden_dim)
        self.decoder = Decoder(d_model=self.decoder_embed_dim, N=decoder_layer, heads=decode_head)
        self.hid2out = nn.Linear(self.seq_hidden_dim, 1)

    def pooling_sum(self, hiddens, lens):
        lens = lens.to(hiddens.device)
        lens = torch.autograd.Variable(torch.unsqueeze(lens, dim=1), requires_grad=False)
        batch_size = range(hiddens.shape[0])
        for i in batch_size:
            hiddens[i, 0] = torch.sum(hiddens[i, :lens[i]], dim=0)
        return hiddens[list(batch_size), 0]

    def seg_embedding(self, x):
        bert_output = self.seg_embedding_learning(
            input_ids=x[0],
            encoder_attention_mask=x[1],
            labels=x[2],
            output_hidden_states=True
        )
        return bert_output["loss"], bert_output["hidden_states"][4], bert_output["logits"]

    def forward(self, inputs, args):
        feature = inputs['links']  # Shape: (batch, seq_len, 25)
        lens = inputs['lens']

        # 1. Feature Slicing
        highwayrep = self.highwayembed(feature[:, :, 0].long())  # Col 0: Highway ID
        # Cols 1..2: seg_len, cum_length
        gpsrep = self.gpsrep(feature[:, :, 3:7])  # Cols 3..6: GPS coords (4)
        context_rep = self.context_mlp(feature[:, :, 7:21])  # Cols 7..20: 14 Global Context features
        dynamic_feats = feature[:, :, 21:25]  # Cols 21..24: 4 Dynamic segment features

        # 2. Masked LM Segment Embeddings
        loss_1, hidden_states, prediction_scores = self.seg_embedding([
            inputs['linkindex'],
            inputs['encoder_attention_mask'],
            inputs['mask_label']
        ])

        # 3. Temporal Context Representation
        timene_input = torch.cat([
            self.seg_embedding_learning.bert.embeddings.word_embeddings(inputs['rawlinks']),
            context_rep
        ], dim=-1)

        timene = self.timene(timene_input) + timene_input

        # 4. Feature Combination
        combined_feats = torch.cat([
            feature[:, :, 1:3],  # seg_len, cum_length (2)
            highwayrep,  # 5
            gpsrep,  # 16
            timene,  # timene_dim
            dynamic_feats  # 4
        ], dim=-1)

        representation = self.represent(combined_feats)
        representation = representation if batch_first else representation.transpose(0, 1).contiguous()

        # 5. GRU Sequence Modeling
        hiddens, rnn_states = self.sequence(representation, seq_lens=lens.long())

        # 6. Decoder & Travel Time Prediction
        decoder = self.decoder(hiddens, lens)
        decoder = decoder if batch_first else decoder.transpose(0, 1).contiguous()
        pooled_decoder = self.pooling_sum(decoder, lens)

        pooled_hidden = torch.cat([pooled_decoder, context_rep[:, 0]], dim=-1)
        hidden = F.leaky_relu(self.input2hid(pooled_hidden))
        output = self.hid2out(hidden)
        output = args.scaler.inverse_transform(output)

        return output, loss_1


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.size = d_model
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)
    if dropout is not None:
        scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.attn_1 = nn.MultiheadAttention(embed_dim=d_model, dropout=dropout, num_heads=self.h)

    def forward(self, q, k, v, len):
        k = self.k_linear(k)
        q = self.q_linear(q)
        v = self.v_linear(v)
        S = q.shape[0]
        mask = torch.stack([torch.cat((torch.zeros(i), torch.ones(S - i)), 0) for i in len]).bool().to(k.device)
        attn_output, attn_output_weights = self.attn_1(q, k, v, key_padding_mask=mask)
        return attn_output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=1024, dropout=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads=1, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x, len):
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, x2, x2, len))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Decoder(nn.Module):
    def __init__(self, d_model, N=3, heads=1, dropout=0.1):
        super().__init__()
        self.N = N
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, x, lens):
        for i in range(self.N):
            x = self.layers[i](x, lens)
        return self.norm(x)