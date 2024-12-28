import torch
import torch.nn as nn
import math
from torch.distributions import Bernoulli


class LSTM_MCDO(nn.Module):
    def __init__(self, input_shape, hidden_rnn, hidden_dense, output_dim, activation):
        super(LSTM_MCDO, self).__init__()
        self.lstm = nn.LSTM(input_size=input_shape, hidden_size=hidden_rnn, batch_first=True)
        self.dense1 = nn.Linear(hidden_rnn, hidden_dense)
        self.activation = nn.ReLU() if activation == 'relu' else nn.Tanh()  # Choose the appropriate activation function.
        self.dropout = nn.Dropout(p=0)
        self.dense2 = nn.Linear(hidden_dense, output_dim)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        
        # print(x.shape, out.shape)
        out = out[:, -1, :]  # Select the output of the last time step.
        out = self.dense1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.dense2(out)
        return out
class TransformerAttn(nn.Module):
    """
    Module that calculates self-attention weights using transformer like attention
    """

    def __init__(self, dim_in=40, value_dim=40, key_dim=40) -> None:
        """
        param dim_in: Dimensionality of input sequence
        param value_dim: Dimension of value transform
        param key_dim: Dimension of key transform
        """
        super(TransformerAttn, self).__init__()
        self.value_layer = nn.Linear(dim_in, value_dim)
        self.query_layer = nn.Linear(dim_in, value_dim)
        self.key_layer = nn.Linear(dim_in, key_dim)

    def forward(self, seq):
        """
        param seq: Sequence in dimension [Seq len, Batch, Hidden size]
        """
        seq_in = seq.transpose(0, 1)
        value = self.value_layer(seq_in)
        query = self.query_layer(seq_in)
        keys = self.key_layer(seq_in)
        weights = (value @ query.transpose(1, 2)) / math.sqrt(seq.shape[-1])
        weights = torch.softmax(weights, -1)
        return (weights @ keys).transpose(1, 0)

    def forward_mask(self, seq, mask):
        """
        param seq: Sequence in dimension [Seq len, Batch, Hidden size]
        """
        seq_in = seq.transpose(0, 1)
        value = self.value_layer(seq_in)
        query = self.query_layer(seq_in)
        keys = self.key_layer(seq_in)
        weights = (value @ query.transpose(1, 2)) / math.sqrt(seq.shape[-1])
        weights = torch.exp(weights)
        weights = (weights.transpose(1, 2) * mask.transpose(1, 0)).transpose(1, 2)
        weights = weights / (weights.sum(-1, keepdim=True))
        return (weights @ keys).transpose(1, 0) * mask
class EmbedAttenSeq(nn.Module):
    """
    Module to embed a sequence. Adds Attention modul
    """

    def __init__(
        self,
        dim_seq_in: int = 5,
        dim_metadata: int = 3,
        rnn_out: int = 40,
        dim_out: int = 50,
        n_layers: int = 1,
        bidirectional: bool = False,
        attn=TransformerAttn,
        dropout=0.0,
    ) -> None:
        """
        param dim_seq_in: Dimensionality of input vector (no. of age groups)
        param dim_out: Dimensionality of output vector
        param dim_metadata: Dimensions of metadata for all sequences
        param rnn_out: output dimension for rnn
        """
        super(EmbedAttenSeq, self).__init__()

        self.dim_seq_in = dim_seq_in
        self.dim_metadata = dim_metadata
        self.rnn_out = rnn_out
        self.dim_out = dim_out
        self.bidirectional = bidirectional

        self.rnn = nn.GRU(
            input_size=self.dim_seq_in,
            hidden_size=self.rnn_out // 2 if self.bidirectional else self.rnn_out,
            bidirectional=bidirectional,
            num_layers=n_layers,
            dropout=dropout,
        )
        self.attn_layer = attn(self.rnn_out, self.rnn_out, self.rnn_out)
        self.out_layer = [
            nn.Linear(
                in_features=self.rnn_out + self.dim_metadata, out_features=self.dim_out
            ),
            nn.Tanh(),
            nn.Dropout(dropout),
        ]
        self.out_layer = nn.Sequential(*self.out_layer)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
        self.out_layer.apply(init_weights)

    def forward_mask(self, seqs, metadata, mask):
        # Take last output from GRU
        latent_seqs = self.rnn(seqs)[0]
        latent_seqs = latent_seqs
        latent_seqs = self.attn_layer.forward_mask(latent_seqs, mask)
        latent_seqs = latent_seqs.sum(0)
        out = self.out_layer(torch.cat([latent_seqs, metadata], dim=1))
        return out

    def forward(self, seqs, metadata=None):
        # Take last output from GRU
        latent_seqs, encoder_hidden = self.rnn(seqs)
        latent_seqs = self.attn_layer(latent_seqs).sum(0)
        if metadata is not None:
            out = self.out_layer(torch.cat([latent_seqs, metadata], dim=1))
        else:
            out = self.out_layer(latent_seqs)
        return out, encoder_hidden
        
class LSTM_Two_Encoder(nn.Module):
    def __init__(self, input_shape, hidden_rnn, hidden_dense, output_dim, activation):
        super(LSTM_Two_Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_shape, hidden_size=hidden_rnn, batch_first=True)
        self.emb_model_2 = EmbedAttenSeq(
            dim_seq_in=1,
            dim_metadata=80,
            rnn_out=32,
            dim_out=32,
            n_layers=2,
            bidirectional=True,
        )
        self.dense1 = nn.Linear(hidden_rnn, hidden_dense)
        self.activation = nn.ReLU() if activation == 'relu' else nn.Tanh()  # Choose the appropriate activation function.
        self.dropout = nn.Dropout(p=0.2)
        self.dense2 = nn.Linear(hidden_dense, output_dim)
        self.alpha = nn.Parameter(torch.zeros(1))
        
    def forward(self, x, t_x, meta):
        
        out, _ = self.lstm(x)
        out_2, _ = self.emb_model_2.forward(t_x.transpose(1, 0), meta)
        out = out[:, -1, :]  # Select the output of the last time step.
        out = out + self.alpha * out_2.sum(dim=0).unsqueeze(0)
        # out = torch.concat([out, out_2.sum(dim=0).unsqueeze(-1)], dim=1)
        out = self.dense1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.dense2(out)
        return out


class LSTM_Three_Encoder(nn.Module):
    def __init__(self, input_shape, hidden_rnn, hidden_dense, output_dim, activation):
        super(LSTM_Three_Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_shape, hidden_size=hidden_rnn, batch_first=True)
        self.emb_model_2 = EmbedAttenSeq(
            dim_seq_in=1,
            dim_metadata=80,
            rnn_out=32,
            dim_out=32,
            n_layers=2,
            bidirectional=True,
        )
        self.lstm_2 = nn.LSTM(input_size=input_shape, hidden_size=hidden_rnn, batch_first=True)
        
        self.dense1 = nn.Linear(hidden_rnn, hidden_dense)
        self.activation = nn.ReLU() if activation == 'relu' else nn.Tanh()  # Choose the appropriate activation function.
        self.dropout = nn.Dropout(p=0.2)
        self.dense2 = nn.Linear(hidden_dense, output_dim)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))
        
    def forward(self, x, t_x, meta):
        case_x = x[:, :1, :]
        other_x = x[:, 1:, :]
        out, _ = self.lstm(case_x)
        out_2, _ = self.emb_model_2.forward(t_x.transpose(1, 0), meta)
        out_3, _ = self.lstm_2(other_x)
        
        out = out[:, -1, :]  # Select the output of the last time step.
        out_3 = out_3.mean(dim=1)
        # print(out.shape, out_2.shape, out_3.shape)
        # input()
        # print(out.shape, out_2.sum(dim=0).shape)
        out = out + 0.01 * self.alpha * out_2.sum(dim=0).unsqueeze(0) + 0.01 * self.beta * out_3
        out = self.dense1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.dense2(out)
        return out