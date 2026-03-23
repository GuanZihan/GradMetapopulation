''' County and State Data Processing network'''

import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.preprocessing import StandardScaler

from data_utils import (
    counties,
    create_window_seqs,
    get_age_group_train_data,
    get_county_train_data,
    get_state_train_data_flu,
    get_time_series_train_data,
)
cuda = torch.device('cuda')
dtype = torch.float
SMOOTH_WINDOW = 7


class MetapopulationSEIRMBeta:
    def __init__(self, params, device, num_patches, migration_matrix, num_agents, seed_infection_status={}):
        super().__init__()
        self.device = device
        self.num_patches = num_patches
        self.state = torch.zeros((num_patches, 5)).to(self.device)
        self.params = params
        self.migration_matrix = migration_matrix.to(self.device)
        self.num_agents = num_agents.to(self.device)
        self.seed_infection_status = seed_infection_status

    def init_compartments(self, learnable_params, seed_infection_status={}):
        ''' let's get initial conditions '''
        initial_infections = torch.zeros((self.num_patches)).to(self.device)
        for state, value in enumerate(seed_infection_status):
            initial_infections[state] = value
        initial_conditions = torch.zeros((self.num_patches, 5)).to(self.device)
        for patch in range(self.num_patches):
            initial_conditions[patch][2] = initial_infections[patch]
            initial_conditions[patch][0] = self.num_agents[patch] - initial_infections[patch]
        self.state = initial_conditions

    def step(self, t, values, seed_status, adjustment_matrix):
        """
        Computes ODE states via equations
            state is the array of state value (S,E,I,R,M) for each patch
        """

        params = {
            'kappa': values[0],
            "symprob": values[1],
            'epsilon': values[2],
            'alpha': values[3],
            'gamma': values[4],
            'delta': values[5],
            'mor': values[6],
            "seed_status": (seed_status).long(),
            "beta_matrix": adjustment_matrix
        }

        if t == 0:
            self.init_compartments(params, seed_infection_status=params["seed_status"])
        
        N_eff = self.migration_matrix.T @ self.num_agents
        I_eff = self.migration_matrix.T @ self.state[:, 2].clone()
        E_eff = self.migration_matrix.T @ self.state[:, 1].clone()

        
        beta_j_eff = I_eff

        beta_j_eff = beta_j_eff / N_eff
        beta_j_eff = beta_j_eff * params["beta_matrix"].mean(dim=0) # modify later 16
        
        beta_j_eff = beta_j_eff * (
            (1 - params["kappa"]) * (1 - params["symprob"]) + params["symprob"]
        )
        beta_j_eff = torch.nan_to_num(beta_j_eff)


        E_beta_j_eff = E_eff
        E_beta_j_eff = E_beta_j_eff / N_eff
        E_beta_j_eff = E_beta_j_eff * params["beta_matrix"].mean(dim=0)
        E_beta_j_eff = E_beta_j_eff * (1 - params["epsilon"])
        E_beta_j_eff = torch.nan_to_num(E_beta_j_eff)


        # Infection force
        beta_sum_eff = beta_j_eff + E_beta_j_eff
        inf_force = self.migration_matrix @ beta_sum_eff

        # New exposures during day t
        new_inf = inf_force * self.state[:, 0].clone()
        new_inf = torch.minimum(new_inf, self.state[:, 0].clone())

        self.state[:, 0] = self.state[:, 0].clone() - new_inf + params["delta"] * self.state[:, 3].clone()
        self.state[:, 1] = new_inf + (1 - params["alpha"]) * self.state[:, 1].clone()
        self.state[:, 2] = params["alpha"] * self.state[:, 1].clone() + (1 - params["gamma"] - params["mor"]) * self.state[:, 2].clone()
        self.state[:, 3] = params["gamma"] * self.state[:, 2].clone() + (1 - params["delta"]) * self.state[:, 3].clone()
        self.state[:, 4] = params["mor"] * self.state[:, 2].clone()

        # (group_id, (time_stamp), state)
        NEW_INFECTIONS_TODAY = self.state[:, 2].clone()
        NEW_DEATHS_TODAY = self.state[:, 4].clone()

        return NEW_DEATHS_TODAY, NEW_INFECTIONS_TODAY
    
    def _compute_Rt_ngm(self, params, N_eff):
        """
        Compute NGM-based R_t using the current state, given parameters and N_eff.

        We use the NGM form:
            K_{i,j}(t) = C_{i,j} * beta_j(t) * S_i(t) / N_eff_j * D
        where:
            - C = migration_matrix
            - beta_j(t) is an effective per-infected transmission rate
            - D = 1 / gamma is mean infectious duration
        """
        # Susceptibles in each patch (current state)
        S = self.state[:, 0]  # shape (num_patches,)

        # Extract scalars as tensors
        kappa = torch.as_tensor(params["kappa"], device=self.device, dtype=torch.float32)
        symprob = torch.as_tensor(params["symprob"], device=self.device, dtype=torch.float32)
        epsilon = torch.as_tensor(params["epsilon"], device=self.device, dtype=torch.float32)
        gamma = torch.as_tensor(params["gamma"], device=self.device, dtype=torch.float32)

        # Mean infectious duration D = 1 / gamma
        D = 1.0 / (gamma + 1e-12)

        # Baseline transmissibility per patch from beta_matrix
        # beta_matrix: (num_time_points, num_patches, ...) or similar; we use mean over dim=0 as in step()
        beta_base = params["beta_matrix"].mean(dim=0).to(self.device)  # shape (num_patches,)

        # Effective weights for I and E contributions (same logic as in step, but per infected)
        beta_I_weight = (1 - kappa) * (1 - symprob) + symprob      # contribution from I
        beta_E_weight = 1 - epsilon                                # contribution from E

        # Effective per-infected transmission rate per patch j
        # This is an approximation that aggregates symptomatic/asymptomatic + pre-symptomatic effects
        beta_j = beta_base * (beta_I_weight + beta_E_weight)       # shape (num_patches,)

        # Build K(t): shape (num_patches, num_patches)
        C = self.migration_matrix                                  # contact / mixing matrix
        S_col = S.view(-1, 1)                                      # (i, 1)
        beta_row = beta_j.view(1, -1)                              # (1, j)
        N_eff_row = N_eff.view(1, -1)                              # (1, j)

        # K_{i,j} = C_{i,j} * beta_j * S_i / N_eff_j * D
        K = C * (S_col * beta_row * D / (N_eff_row + 1e-12))

        # Eigenvalues and R_t as dominant eigenvalue (spectral radius)
        eigvals = torch.linalg.eigvals(K)      # complex tensor
        Rt = eigvals.real.max()               # maximum real part

        return Rt, K, eigvals

    def compute_Rt(self, values, adjustment_matrix):
        """
        Public method to compute R_t at the *current* state using the same
        parameter vector `values` and `adjustment_matrix` as in `step`.

        Call this AFTER you've updated `self.state` (e.g., after `step`).
        """
        params = {
            'kappa': values[0],
            "symprob": values[1],
            'epsilon': values[2],
            'alpha': values[3],
            'gamma': values[4],
            'delta': values[5],
            'mor': values[6],
            "beta_matrix": adjustment_matrix
        }

        # Effective total population per patch (same N_eff as in step, but only depends on num_agents)
        N_eff = self.migration_matrix.T @ self.num_agents          # shape (num_patches,)

        Rt, K, eigvals = self._compute_Rt_ngm(params, N_eff)
        return Rt, K, eigvals

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


class DecodeSeq(nn.Module):
    """
    Module to embed a sequence. Adds Attention modul
    """

    def __init__(
        self,
        dim_seq_in: int = 5,
        dim_metadata: int = 3,
        rnn_out: int = 40,
        dim_out: int = 5,
        n_layers: int = 1,
        bidirectional: bool = False,
        dropout=0.0,
    ) -> None:
        """
        param dim_seq_in: Dimensionality of input vector (no. of age groups)
        param dim_out: Dimensionality of output vector
        param dim_metadata: Dimensions of metadata for all sequences
        param rnn_out: output dimension for rnn
        """
        super(DecodeSeq, self).__init__()

        self.dim_seq_in = dim_seq_in
        self.dim_metadata = dim_metadata
        self.rnn_out = rnn_out
        self.dim_out = dim_out
        self.bidirectional = bidirectional

        self.act_fcn = nn.Tanh()

        # to embed input
        self.embed_input = nn.Linear(self.dim_seq_in, self.rnn_out) 

        # to combine input and context
        self.attn_combine = nn.Linear(2*self.rnn_out, self.rnn_out)

        self.rnn = nn.GRU(
            input_size=self.rnn_out,
            hidden_size=self.rnn_out // 2 if self.bidirectional else self.rnn_out,
            bidirectional=bidirectional,
            num_layers=n_layers,
            dropout=dropout,
        )
        self.out_layer = [
            nn.Linear(
                in_features=self.rnn_out, out_features=self.dim_out
            ),
            nn.Tanh(),
            nn.Dropout(dropout),
        ]
        self.out_layer = nn.Sequential(*self.out_layer)

        # initialize
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
        self.out_layer.apply(init_weights)
        self.embed_input.apply(init_weights)
        self.attn_combine.apply(init_weights)

    def forward(self, Hi_data, encoder_hidden, context):
        # Hi_data is scaled time
        inputs = Hi_data.transpose(1,0)
        if self.bidirectional:
            h0 = encoder_hidden[2:]
        else:
            h0 = encoder_hidden[2:].sum(0).unsqueeze(0)
        # combine input and context
        inputs = self.embed_input(inputs)
        # repeat context for each item in sequence  
        context = context.repeat(inputs.shape[0],1,1)
        inputs = torch.cat((inputs, context), 2) 
        inputs = self.attn_combine(inputs)
        # Take last output from GRU
        
        latent_seqs = self.rnn(inputs, h0)[0]
        latent_seqs = latent_seqs.transpose(1,0)
        latent_seqs = self.out_layer(latent_seqs)
        return latent_seqs


''' smooth data with moving average (common with fitting mechanistic models) '''
def moving_average(x, w):
    return pd.Series(x).rolling(w, min_periods=1).mean().values


''' smooth data with moving average (common with fitting mechanistic models) '''
def fetch_county_data_covid(state='MA', county_id='25005', pred_week='202021', batch_size=32, noise_level=0, args=None):
    ''' Import COVID data for counties '''
    np.random.seed(17)

    if county_id == 'all':
        all_counties = counties[state]
    else:
        all_counties = [county_id]

    c_seqs = []  # county sequences of features
    c_ys = []  # county targets
    for county in all_counties:
        X_county, y = get_county_train_data(county,pred_week,noise_level=noise_level, args=args)
        y = moving_average(y[:,1].ravel(),SMOOTH_WINDOW).reshape(-1,1)
        c_seqs.append(X_county.to_numpy())
        c_ys.append(y)
    

    
    c_seqs = np.array(c_seqs).sum(0)  # Shape: [regions, time, features]

    c_ys = np.array(c_ys).sum(0)  # Shape: [regions, time, 1]

    # Normalize
    # One scaler per county
    scalers = [StandardScaler() for _ in range(len(all_counties))]
    c_seqs_norm = []
    for i, scaler in enumerate(scalers):
        c_seqs_norm.append(scaler.fit_transform(c_seqs[i]))
    c_seqs_norm = np.array(c_seqs_norm)

    ''' Create static metadata data for each county '''

    county_idx = {r: i for i, r in enumerate(all_counties)}
    def one_hot(idx, dim=len(county_idx)):
        ans = np.zeros(dim, dtype="float32")
        ans[idx] = 1.0
        return ans
    metadata = np.array([one_hot(county_idx[r]) for r in all_counties])

    ''' Prepare train and validation dataset '''

    min_sequence_length = 20
    metas, seqs, y, y_mask = [], [], [], []
    for meta, seq, ys in zip(metadata, c_seqs_norm, c_ys):
        seq, ys, ys_mask = create_window_seqs(seq,ys,min_sequence_length)
        metas.append(meta)
        seqs.append(seq[[-1]])
        y.append(ys[[-1]])
        y_mask.append(ys_mask[[-1]])

    all_metas = np.array(metas, dtype="float32")
    all_county_seqs = torch.cat(seqs,axis=0)
    all_county_ys = torch.cat(y,axis=0)
    all_county_y_mask = torch.cat(y_mask,axis=0)
    
    counties_train, metas_train, X_train, y_train, y_mask_train = \
        all_counties, all_metas, all_county_seqs, all_county_ys, all_county_y_mask
    
    train_dataset = SeqData(counties_train, metas_train, X_train, y_train, y_mask_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    assert all_county_seqs.shape[1] == all_county_ys.shape[1]
    seqlen = all_county_seqs.shape[1]
    return train_loader, metas_train.shape[1], X_train.shape[2], seqlen


def fetch_time_series_data(batch_size=32, noise_level=0, args=None, split="train"):
    ''' Import COVID data for counties '''
    ''' Import COVID data for counties '''
    np.random.seed(17)
    all_features = ["GHT"]


    c_seqs = []  # county sequences of features
    c_ys = []  # county targets
    for feature in all_features:
        X_county, y = get_time_series_train_data(feature,noise_level=noise_level, args=args)
        y = moving_average(y[:,1].ravel(),SMOOTH_WINDOW).reshape(-1,1)
        # y = y[:, 1].reshape(-1, 1)
        c_seqs.append(X_county.to_numpy())
        c_ys.append(y)
        print(X_county.shape)
        input()
    c_seqs = np.array(c_seqs)  # Shape: [regions, time, features]

    c_ys = np.array(c_ys)  # Shape: [regions, time, 1]

    # Normalize
    # One scaler per county
    scalers = [StandardScaler() for _ in range(len(all_features))]
    c_seqs_norm = []
    for i, scaler in enumerate(scalers):
        c_seqs_norm.append(scaler.fit_transform(c_seqs[i]))
    c_seqs_norm = np.array(c_seqs_norm)

    ''' Create static metadata data for each county '''

    county_idx = {r: i for i, r in enumerate(all_features)}
    def one_hot(idx, dim=len(county_idx)):
        ans = np.zeros(dim, dtype="float32")
        ans[idx] = 1.0
        return ans
    metadata = np.array([one_hot(county_idx[r]) for r in all_features])

    ''' Prepare train and validation dataset '''

    min_sequence_length = 20
    metas, seqs, y, y_mask = [], [], [], []
    for meta, seq, ys in zip(metadata, c_seqs_norm, c_ys):
        seq, ys, ys_mask = create_window_seqs(seq,ys,min_sequence_length)
        metas.append(meta)
        seqs.append(seq[[-1]])
        y.append(ys[[-1]])
        y_mask.append(ys_mask[[-1]])

    all_metas = np.array(metas, dtype="float32")
    all_county_seqs = torch.cat(seqs,axis=0)
    all_county_ys = torch.cat(y,axis=0)
    all_county_y_mask = torch.cat(y_mask,axis=0)
    
    counties_train, metas_train, X_train, y_train, y_mask_train = \
        all_features, all_metas, all_county_seqs, all_county_ys, all_county_y_mask
    
    train_dataset = SeqData(counties_train, metas_train, X_train, y_train, y_mask_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    

    assert all_county_seqs.shape[1] == all_county_ys.shape[1]
    seqlen = all_county_seqs.shape[1]
    return train_loader, metas_train.shape[1], X_train.shape[2], seqlen
''' smooth data with moving average (common with fitting mechanistic models) '''
def fetch_age_group_data_covid(batch_size=32, noise_level=0, args=None, split="train"):
    ''' Import COVID data for counties '''
    np.random.seed(17)
    all_age_groups = ["00-05", "05-10", "10-15", "15-20", "20-25", "25-30", "30-35", "35-40", "40-45", "45-50", "50-55", "55-60", "60-65", "65-70", "70-75", "75+"]


    c_seqs = []  # county sequences of features
    c_ys = []  # county targets
    for age in all_age_groups:
        X_county, y = get_age_group_train_data(age,noise_level=noise_level, args=args, split=split)
        y = moving_average(y[:,1].ravel(),SMOOTH_WINDOW).reshape(-1,1)
        # y = y[:, 1].reshape(-1, 1)
        c_seqs.append(X_county.to_numpy())
        c_ys.append(y)
    c_seqs = np.array(c_seqs)  # Shape: [regions, time, features]

    c_ys = np.array(c_ys)  # Shape: [regions, time, 1]

    # Normalize
    # One scaler per county
    scalers = [StandardScaler() for _ in range(len(all_age_groups))]
    c_seqs_norm = []
    for i, scaler in enumerate(scalers):
        c_seqs_norm.append(scaler.fit_transform(c_seqs[i]))
    c_seqs_norm = np.array(c_seqs_norm)

    ''' Create static metadata data for each county '''

    county_idx = {r: i for i, r in enumerate(all_age_groups)}
    def one_hot(idx, dim=len(county_idx)):
        ans = np.zeros(dim, dtype="float32")
        ans[idx] = 1.0
        return ans
    metadata = np.array([one_hot(county_idx[r]) for r in all_age_groups])

    ''' Prepare train and validation dataset '''

    min_sequence_length = 20
    metas, seqs, y, y_mask = [], [], [], []
    for meta, seq, ys in zip(metadata, c_seqs_norm, c_ys):
        seq, ys, ys_mask = create_window_seqs(seq,ys,min_sequence_length)
        metas.append(meta)
        seqs.append(seq[[-1]])
        y.append(ys[[-1]])
        y_mask.append(ys_mask[[-1]])

    all_metas = np.array(metas, dtype="float32")
    all_county_seqs = torch.cat(seqs,axis=0)
    all_county_ys = torch.cat(y,axis=0)
    all_county_y_mask = torch.cat(y_mask,axis=0)
    
    counties_train, metas_train, X_train, y_train, y_mask_train = \
        all_age_groups, all_metas, all_county_seqs, all_county_ys, all_county_y_mask
    
    train_dataset = SeqData(counties_train, metas_train, X_train, y_train, y_mask_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    

    assert all_county_seqs.shape[1] == all_county_ys.shape[1]
    seqlen = all_county_seqs.shape[1]
    return train_loader, metas_train.shape[1], X_train.shape[2], seqlen



def fetch_county_data_flu(state='MA', county_id='25005', pred_week='202021', batch_size=32, noise_level=0):
    ''' in flu, our features are state-level and target ILI is only available at state'''
    np.random.seed(17)

    ''' Import data for all counties '''

    if county_id == 'all':
        all_counties = counties[state]
    else:
        all_counties = [county_id]

    X_state, y = get_state_train_data_flu(state, pred_week, noise_level=noise_level)
    y = moving_average(y.ravel(),SMOOTH_WINDOW).reshape(-1,1)
    c_seqs = []  # county sequences of features
    c_ys = []  # county targets
    for _ in all_counties:
        c_seqs.append(X_state.to_numpy())
        c_ys.append(y)
    c_seqs = np.array(c_seqs)  # Shape: [regions, time, features]
    c_ys = np.array(c_ys)  # Shape: [regions, time, 1]

    # Normalize
    # One scaler per county
    scalers = [StandardScaler() for _ in range(len(all_counties))]
    c_seqs_norm = []
    for i, scaler in enumerate(scalers):
        c_seqs_norm.append(scaler.fit_transform(c_seqs[i]))
    c_seqs_norm = np.array(c_seqs_norm)

    ''' Create static metadata data for each county '''

    county_idx = {r: i for i, r in enumerate(all_counties)}
    def one_hot(idx, dim=len(county_idx)):
        ans = np.zeros(dim, dtype="float32")
        ans[idx] = 1.0
        return ans
    metadata = np.array([one_hot(county_idx[r]) for r in all_counties])

    ''' Prepare train and validation dataset '''
    min_sequence_length = 5
    metas, seqs, y, y_mask = [], [], [], []
    for meta, seq, ys in zip(metadata, c_seqs_norm, c_ys):
        seq, ys, ys_mask = create_window_seqs(seq,ys,min_sequence_length)
        metas.append(meta)
        seqs.append(seq[[-1]])
        y.append(ys[[-1]])
        y_mask.append(ys_mask[[-1]])

    all_metas = np.array(metas, dtype="float32")
    all_county_seqs = torch.cat(seqs,axis=0)
    all_county_ys = torch.cat(y,axis=0)
    all_county_y_mask = torch.cat(y_mask,axis=0)
    
    counties_train, metas_train, X_train, y_train, y_mask_train = \
        all_counties, all_metas, all_county_seqs, all_county_ys, all_county_y_mask
    train_dataset = SeqData(counties_train, metas_train, X_train, y_train, y_mask_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    assert all_county_seqs.shape[1] == all_county_ys.shape[1]
    seqlen = all_county_seqs.shape[1]

    return train_loader, metas_train.shape[1], X_train.shape[2], seqlen

# dataset class
class SeqData(torch.utils.data.Dataset):
    def __init__(self, region, meta, X, y, mask_y):
        self.region = region
        self.meta = meta
        self.X = X
        self.y = y
        # self.mask_y = mask_y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (
            self.region[idx],
            self.meta[idx],
            self.X[idx, :, :],
            self.y[idx]
        )

class ODE(nn.Module):
    def __init__(self, params, device):
        super(ODE, self).__init__()
        county_id = params['county_id']
        abm_params = f'Data/{county_id}_generated_params.yaml'
        #Reading params
        with open(abm_params, 'r') as stream:
            try:
                abm_params = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print('Error in reading parameters file')
                print(exc)
        params.update(abm_params)
        self.params = params
        self.device = device
        self.num_agents = self.params['num_agents'] # Population


class SEIRM(ODE):
    def __init__(self, params, device):
        super().__init__(params,device)
    
    def init_compartments(self,learnable_params):
        ''' let's get initial conditions '''
        initial_infections_percentage = learnable_params['initial_infections_percentage']
        initial_conditions = torch.empty((5)).to(self.device)
        no_infected = (initial_infections_percentage / 100) * self.num_agents # 1.0 is ILI
        initial_conditions[2] = no_infected
        initial_conditions[0] = self.num_agents - no_infected
        print('initial infected',no_infected)
        self.state = initial_conditions

    def step(self, t, values):
        """
        Computes ODE states via equations       
            state is the array of state value (S,E,I,R,M)
        """
        params = {
            'beta':values[0],
            'alpha':values[1],
            'gamma':values[2],
            'mu':values[3],
            'initial_infections_percentage': values[4], 
        }
        if t==0:
            self.init_compartments(params)
        # to make the NN predict lower numbers, we can make its prediction to be N-Susceptible
        dSE = params['beta'] * self.state[0] * self.state[2] / self.num_agents
        dEI = params['alpha'] * self.state[1]
        dIR = params['gamma'] * self.state[2]
        dIM = params['mu'] * self.state[2]

        dS  = -1.0 * dSE
        dE  = dSE - dEI
        dI = dEI - dIR - dIM
        dR  = dIR
        dM  = dIM

        # concat and reshape to make it rows as obs, cols as states
        self.dstate = torch.stack([dS, dE, dI, dR, dM], 0)
        NEW_INFECTIONS_TODAY = dEI
        NEW_DEATHS_TODAY = dIM
        # update state
        self.state = self.state + self.dstate
        
        return NEW_INFECTIONS_TODAY, NEW_DEATHS_TODAY

class SIRS(ODE):
    def __init__(self, params, device):
        super().__init__(params,device)

    def init_compartments(self,learnable_params):
        ''' let's get initial conditions '''
        initial_infections_percentage = learnable_params['initial_infections_percentage']
        initial_conditions = torch.empty((2)).to(self.device)
        no_infected = (initial_infections_percentage / 100) * self.num_agents # 1.0 is ILI
        initial_conditions[1] = no_infected
        initial_conditions[0] = self.num_agents - no_infected
        print('initial infected',no_infected)

        self.state = initial_conditions

    def step(self, t, values):
        """
        Computes ODE states via equations       
            state is the array of state value (S,I)
        """
        params = {
            'beta':values[0],  # contact rate, range: 0-1
            'initial_infections_percentage': values[1], 
        }
        # set from expertise
        params['D'] = 3.5
        params['L'] = 2000
        if t==0:
            self.init_compartments(params)
        dS =  (self.num_agents - self.state[0] - self.state[1]) / params['L'] -  params['beta'] * self.state[0] * self.state[1] / self.num_agents
        dSI = params['beta'] * self.state[0] * self.state[1] / self.num_agents
        dI = dSI - self.state[1] / params['D']

        # concat and reshape to make it rows as obs, cols as states
        self.dstate = torch.stack([dS, dI], 0)

        NEW_INFECTIONS_TODAY = dSI
        # ILI is percentage of outpatients with influenza-like illness
        # ILI = params['lambda'] * dSI / self.num_agents
        # this is what Shaman and Pei do https://github.com/SenPei-CU/Multi-Pathogen_ILI_Forecast/blob/master/code/SIRS_AH.m
        ILI =  dSI / self.num_agents * 100 # multiply 100 because it is percentage

        # update state
        self.state = self.state + self.dstate
        return NEW_INFECTIONS_TODAY, ILI

if __name__ == '__main__':
    print("THIS SHOULD NOT EXECUTE!")
    """ Create model """