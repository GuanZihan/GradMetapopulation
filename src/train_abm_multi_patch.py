import random
import numpy as np
import torch
import os
import torch.nn as nn
import math
import time
from torch.autograd import Variable
from data_utils import WEEKS_AHEAD, states, counties
from copy import copy
import matplotlib.pyplot as plt
from model_utils import EmbedAttenSeq, fetch_county_data_covid, fetch_county_data_flu, DecodeSeq, SEIRM, SIRS, MetapopulationSEIRM, fetch_age_group_data_covid, moving_average
import pdb
import opendp
# from dp_optimizer import DPOptimizer

BENCHMARK_TRAIN = False
NUM_EPOCHS_DIFF = 500
print("---- MAIN IMPORTS SUCCESSFUL -----")
# python -u main.py -st MN -c 27109 -d 1 -ew 202210 --seed 1234 -m ABM-expert -di COVID --exp 601

CONFIGS = {
    "bogota": {
        "num_patch": 16,
        "learning_rate": 1e-4,
        "train_days": 39*7,
        "test_days": 28,
        "num_pub_features": 4
    },
    "COVID": {
        "num_patch": 10,
        "learning_rate": 1e-4,
        "train_days": 175,
        "test_days": 28
    }
}




DAYS_HEAD = 4*7  # 4 weeks ahead

pi = torch.FloatTensor([math.pi])

SAVE_MODEL_PATH = './Models/'

# neural network predicting parameters of the ABM

class CalibNN(nn.Module):
    def __init__(self, params, metas_train_dim, X_train_dim, device, training_weeks, hidden_dim=32, out_dim=1, n_layers=2, scale_output='abm-covid', bidirectional=True):
        super().__init__()

        self.device = device

        self.training_weeks = training_weeks

        self.params = params

        ''' tune '''
        hidden_dim=64
        out_layer_dim = 32
        
        self.emb_model = EmbedAttenSeq(
            dim_seq_in=X_train_dim,
            dim_metadata=metas_train_dim,
            rnn_out=hidden_dim,
            dim_out=hidden_dim,
            n_layers=n_layers,
            bidirectional=bidirectional,
        ) 

        self.decoder = DecodeSeq(
            dim_seq_in=1,
            rnn_out=hidden_dim, # divides by 2 if bidirectional
            dim_out=out_layer_dim,
            n_layers=1,
            bidirectional=True,
        ) 

        out_layer_width = out_layer_dim
        self.out_layer =  [
            nn.Linear(
                in_features=out_layer_width, out_features=out_layer_width//2
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=out_layer_width//2, out_features=out_dim
            ),
        ]
        self.out_layer = nn.Sequential(*self.out_layer)

        self.out_layer2 =  [
            nn.Linear(
                in_features=out_layer_width, out_features=out_layer_width//2
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=out_layer_width//2, out_features=CONFIGS[self.params['disease']]["num_patch"]
            ),
        ]
        self.out_layer2 = nn.Sequential(*self.out_layer2)


        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
        self.out_layer.apply(init_weights)
        self.out_layer2.apply(init_weights)
        self.min_values = torch.tensor(MIN_VAL_PARAMS[scale_output],device=self.device)
        self.max_values = torch.tensor(MAX_VAL_PARAMS[scale_output],device=self.device)
        self.min_values_2 = torch.tensor(MIN_VAL_PARAMS_2[scale_output],device=self.device)
        self.max_values_2 = torch.tensor(MAX_VAL_PARAMS_2[scale_output],device=self.device)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, meta):
        x_embeds, encoder_hidden = self.emb_model.forward(x.transpose(1, 0), meta)
        # create input that will tell the neural network which week it is predicting
        # thus, we have one element in the sequence per each week of R0
        time_seq = torch.arange(1,self.training_weeks+WEEKS_AHEAD+1).repeat(x_embeds.shape[0],1).unsqueeze(2)
        Hi_data = ((time_seq - time_seq.min())/(time_seq.max() - time_seq.min())).to(self.device)
        emb = self.decoder(Hi_data, encoder_hidden, x_embeds) 
        out = self.out_layer(emb)
        out = torch.mean(out, dim=0)
        out = self.min_values + (self.max_values-self.min_values)*self.sigmoid(out) # (175, 5)

        emb_mean = torch.mean(emb, dim=0)
        # print(emb_mean.shape)
        # emb_mean = torch.mean(emb_mean, dim=0)
        emb_mean = emb_mean[-1, :]
        
        out2 = self.out_layer2(emb_mean)
        # print(out2.shape)
        # input()
        out2 = self.min_values_2 + (self.max_values_2-self.min_values_2)*self.sigmoid(out2) # (5)
        return out, out2


class CalibNNThreeOutputs(nn.Module):
    def __init__(self, params, metas_train_dim, X_train_dim, device, training_weeks, hidden_dim=32, out_dim=1, n_layers=2, scale_output='abm-covid', bidirectional=True):
        super().__init__()

        self.device = device

        self.training_weeks = training_weeks

        self.params = params

        ''' tune '''
        hidden_dim=64
        out_layer_dim = 32
        
        self.emb_model = EmbedAttenSeq(
            dim_seq_in=X_train_dim,
            dim_metadata=metas_train_dim,
            rnn_out=hidden_dim,
            dim_out=hidden_dim,
            n_layers=n_layers,
            bidirectional=bidirectional,
        ) 

        self.decoder = DecodeSeq(
            dim_seq_in=1,
            rnn_out=hidden_dim, # divides by 2 if bidirectional
            dim_out=out_layer_dim,
            n_layers=1,
            bidirectional=True,
        ) 

        out_layer_width = out_layer_dim
        self.out_layer =  [
            nn.Linear(
                in_features=out_layer_width, out_features=out_layer_width//2
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=out_layer_width//2, out_features=out_dim
            ),
        ]
        self.out_layer = nn.Sequential(*self.out_layer)

        self.out_layer2 =  [
            nn.Linear(
                in_features=out_layer_width, out_features=out_layer_width//2
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=out_layer_width//2, out_features=CONFIGS[self.params['disease']]["num_patch"]
            ),
        ]
        self.out_layer2 = nn.Sequential(*self.out_layer2)

        self.out_layer3 =  [
            nn.Linear(
                in_features=out_layer_width, out_features=out_layer_width//2
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=out_layer_width//2, out_features=CONFIGS[self.params['disease']]["num_patch"]
            ),
        ]
        self.out_layer3 = nn.Sequential(*self.out_layer3)


        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
        self.out_layer.apply(init_weights)
        self.out_layer2.apply(init_weights)
        self.out_layer3.apply(init_weights)
        self.min_values = torch.tensor(MIN_VAL_PARAMS[scale_output],device=self.device)
        self.max_values = torch.tensor(MAX_VAL_PARAMS[scale_output],device=self.device)
        self.min_values_2 = torch.tensor(MIN_VAL_PARAMS_2[scale_output],device=self.device)
        self.max_values_2 = torch.tensor(MAX_VAL_PARAMS_2[scale_output],device=self.device)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, meta):
        x_embeds, encoder_hidden = self.emb_model.forward(x.transpose(1, 0), meta)
        # create input that will tell the neural network which week it is predicting
        # thus, we have one element in the sequence per each week of R0
        time_seq = torch.arange(1,self.training_weeks+WEEKS_AHEAD+1).repeat(x_embeds.shape[0],1).unsqueeze(2)
        Hi_data = ((time_seq - time_seq.min())/(time_seq.max() - time_seq.min())).to(self.device)
        emb = self.decoder(Hi_data, encoder_hidden, x_embeds) 
        out = self.out_layer(emb)
        out = torch.mean(out, dim=0)
        out = self.min_values + (self.max_values-self.min_values)*self.sigmoid(out) # (175, 5)

        emb_mean = torch.mean(emb, dim=0)
        # print(emb_mean.shape)
        # emb_mean = torch.mean(emb_mean, dim=0)
        emb_mean = emb_mean[-1, :]
        
        out2 = self.out_layer2(emb_mean)
        out2 = self.min_values_2 + (self.max_values_2-self.min_values_2)*self.sigmoid(out2) # (5)

        out3 = -self.sigmoid(self.out_layer3(emb_mean))
        # out3 = self.sigmoid(out3) * 0.0004 - 0.0002
        
        # print(torch.sum(out3[:, 0]))
        # input()

        return out, out2, out3
class CalibNNTwoEncoderThreeOutputs(nn.Module):
    def __init__(self, params, metas_train_dim, X_train_dim, device, training_weeks, hidden_dim=32, out_dim=1, n_layers=2, scale_output='abm-covid', bidirectional=True):
        super().__init__()

        self.device = device

        self.training_weeks = training_weeks

        self.params = params

        ''' tune '''
        hidden_dim=64
        out_layer_dim = 32
        
        self.emb_model_1 = EmbedAttenSeq(
            dim_seq_in=X_train_dim,
            dim_metadata=metas_train_dim,
            rnn_out=hidden_dim,
            dim_out=hidden_dim,
            n_layers=n_layers,
            bidirectional=bidirectional,
        )

        self.emb_model_2 = EmbedAttenSeq(
            dim_seq_in=CONFIGS[self.params["disease"]]["num_pub_features"],
            dim_metadata=0,
            rnn_out=hidden_dim,
            dim_out=hidden_dim,
            n_layers=n_layers,
            bidirectional=bidirectional,
        )

        self.decoder = DecodeSeq(
            dim_seq_in=1,
            rnn_out=hidden_dim, # divides by 2 if bidirectional
            dim_out=out_layer_dim,
            n_layers=1,
            bidirectional=True,
        ) 

        out_layer_width = out_layer_dim
        self.out_layer =  [
            nn.Linear(
                in_features=out_layer_width, out_features=out_layer_width//2
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=out_layer_width//2, out_features=out_dim
            ),
        ]
        self.out_layer = nn.Sequential(*self.out_layer)

        self.out_layer2 =  [
            nn.Linear(
                in_features=out_layer_width, out_features=out_layer_width//2
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=out_layer_width//2, out_features=CONFIGS[self.params['disease']]["num_patch"]
            ),
        ]
        self.out_layer2 = nn.Sequential(*self.out_layer2)

        self.out_layer3 =  [
            nn.Linear(
                in_features=out_layer_width, out_features=out_layer_width//2
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=out_layer_width//2, out_features=CONFIGS[self.params['disease']]["num_patch"]*CONFIGS[self.params['disease']]["num_patch"]
            ),
        ]
        self.out_layer3 = nn.Sequential(*self.out_layer3)


        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
        self.out_layer.apply(init_weights)
        self.out_layer2.apply(init_weights)
        self.out_layer3.apply(init_weights)
        self.min_values = torch.tensor(MIN_VAL_PARAMS[scale_output],device=self.device)
        self.max_values = torch.tensor(MAX_VAL_PARAMS[scale_output],device=self.device)
        self.min_values_2 = torch.tensor(MIN_VAL_PARAMS_2[scale_output],device=self.device)
        self.max_values_2 = torch.tensor(MAX_VAL_PARAMS_2[scale_output],device=self.device)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, meta, x_2):
        x_embeds, encoder_hidden = self.emb_model_1.forward(x.transpose(1, 0), meta)
        x_embeds_2, encoder_hidden_2 = self.emb_model_2.forward(x_2.transpose(1, 0))
        # print(x_2)
        # input()
        # print(x_embeds.shape, x_embeds_2.shape)
        # print(encoder_hidden.shape, encoder_hidden_2.shape)
        # input()
        x_embeds = torch.cat([x_embeds, x_embeds_2], dim=0)
        encoder_hidden = torch.cat([encoder_hidden, encoder_hidden_2], dim=1)
        # create input that will tell the neural network which week it is predicting
        # thus, we have one element in the sequence per each week of R0
        time_seq = torch.arange(1,self.training_weeks+WEEKS_AHEAD+1).repeat(x_embeds.shape[0],1).unsqueeze(2)
        Hi_data = ((time_seq - time_seq.min())/(time_seq.max() - time_seq.min())).to(self.device)
        emb = self.decoder(Hi_data, encoder_hidden, x_embeds) 
        out = self.out_layer(emb)
        out = torch.mean(out, dim=0)
        out = self.min_values + (self.max_values-self.min_values)*self.sigmoid(out) # (175, 5)

        emb_mean = torch.mean(emb, dim=0)
        # print(emb_mean.shape)
        # emb_mean = torch.mean(emb_mean, dim=0)
        emb_mean = emb_mean[-1, :]
        
        out2 = self.out_layer2(emb_mean)
        # print(out2.shape)
        # input()
        out2 = self.min_values_2 + (self.max_values_2-self.min_values_2)*self.sigmoid(out2) # (5)

        out3 = self.sigmoid(self.out_layer3(emb_mean).reshape((CONFIGS[self.params['disease']]["num_patch"], CONFIGS[self.params['disease']]["num_patch"])))*2 - 1
        return out, out2, out3



class CalibNNThreeEncoderThreeOutputs(nn.Module):
    def __init__(self, params, metas_train_dim, X_train_dim, device, training_weeks, hidden_dim=32, out_dim=1, n_layers=2, scale_output='abm-covid', bidirectional=True):
        super().__init__()

        self.device = device

        self.training_weeks = training_weeks

        self.params = params

        ''' tune '''
        hidden_dim=64
        out_layer_dim = 32
        
        self.emb_model_1 = EmbedAttenSeq(
            dim_seq_in=X_train_dim,
            dim_metadata=metas_train_dim,
            rnn_out=hidden_dim,
            dim_out=hidden_dim,
            n_layers=n_layers,
            bidirectional=bidirectional,
        )

        self.emb_model_2 = EmbedAttenSeq(
            dim_seq_in=CONFIGS[self.params["disease"]]["num_pub_features"],
            dim_metadata=0,
            rnn_out=hidden_dim,
            dim_out=hidden_dim,
            n_layers=n_layers,
            bidirectional=bidirectional,
        )

        self.emb_model_3 = EmbedAttenSeq(
            dim_seq_in=1,
            dim_metadata=80,
            rnn_out=hidden_dim,
            dim_out=hidden_dim,
            n_layers=n_layers,
            bidirectional=bidirectional,
        )

        self.decoder = DecodeSeq(
            dim_seq_in=1,
            rnn_out=hidden_dim, # divides by 2 if bidirectional
            dim_out=out_layer_dim,
            n_layers=1,
            bidirectional=True,
        ) 

        out_layer_width = out_layer_dim
        self.out_layer =  [
            nn.Linear(
                in_features=out_layer_width, out_features=out_layer_width//2
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=out_layer_width//2, out_features=out_dim
            ),
        ]
        self.out_layer = nn.Sequential(*self.out_layer)

        self.out_layer2 =  [
            nn.Linear(
                in_features=out_layer_width, out_features=out_layer_width//2
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=out_layer_width//2, out_features=CONFIGS[self.params['disease']]["num_patch"]
            ),
        ]
        self.out_layer2 = nn.Sequential(*self.out_layer2)

        self.out_layer3 =  [
            nn.Linear(
                in_features=out_layer_width, out_features=out_layer_width//2
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=out_layer_width//2, out_features=CONFIGS[self.params['disease']]["num_patch"]*CONFIGS[self.params['disease']]["num_patch"]
            ),
        ]
        self.out_layer3 = nn.Sequential(*self.out_layer3)


        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
        self.out_layer.apply(init_weights)
        self.out_layer2.apply(init_weights)
        self.out_layer3.apply(init_weights)
        self.min_values = torch.tensor(MIN_VAL_PARAMS[scale_output],device=self.device)
        self.max_values = torch.tensor(MAX_VAL_PARAMS[scale_output],device=self.device)
        self.min_values_2 = torch.tensor(MIN_VAL_PARAMS_2[scale_output],device=self.device)
        self.max_values_2 = torch.tensor(MAX_VAL_PARAMS_2[scale_output],device=self.device)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, meta, x_2, transaction_x, meta2):
        x_embeds, encoder_hidden = self.emb_model_1.forward(x.transpose(1, 0), meta)
        x_embeds_2, encoder_hidden_2 = self.emb_model_2.forward(x_2.transpose(1, 0))
        x_embeds_3, encoder_hidden_3 = self.emb_model_3.forward(transaction_x.transpose(1, 0), meta2)
        

        x_embeds = torch.cat([x_embeds, x_embeds_2, x_embeds_3], dim=0)
        encoder_hidden = torch.cat([encoder_hidden, encoder_hidden_2, encoder_hidden_3], dim=1)
        # create input that will tell the neural network which week it is predicting
        # thus, we have one element in the sequence per each week of R0
        time_seq = torch.arange(1,self.training_weeks+WEEKS_AHEAD+1).repeat(x_embeds.shape[0],1).unsqueeze(2)
        Hi_data = ((time_seq - time_seq.min())/(time_seq.max() - time_seq.min())).to(self.device)
        emb = self.decoder(Hi_data, encoder_hidden, x_embeds) 
        out = self.out_layer(emb)
        out = torch.mean(out, dim=0)
        out = self.min_values + (self.max_values-self.min_values)*self.sigmoid(out) # (175, 5)

        emb_mean = torch.mean(emb, dim=0)
        # print(emb_mean.shape)
        # emb_mean = torch.mean(emb_mean, dim=0)
        emb_mean = emb_mean[-1, :]
        
        out2 = self.out_layer2(emb_mean)
        # print(out2.shape)
        # input()
        out2 = self.min_values_2 + (self.max_values_2-self.min_values_2)*self.sigmoid(out2) # (5)

        out3 = self.sigmoid(self.out_layer3(emb_mean).reshape((CONFIGS[self.params['disease']]["num_patch"], CONFIGS[self.params['disease']]["num_patch"])))*2 - 1
        return out, out2, out3


class CalibNNTwoEncoder(nn.Module):
    def __init__(self, params, metas_train_dim, X_train_dim, device, training_weeks, hidden_dim=32, out_dim=1, n_layers=2, scale_output='abm-covid', bidirectional=True):
        super().__init__()

        self.device = device

        self.training_weeks = training_weeks

        self.params = params

        ''' tune '''
        hidden_dim=64
        out_layer_dim = 32
        
        self.emb_model_1 = EmbedAttenSeq(
            dim_seq_in=X_train_dim,
            dim_metadata=metas_train_dim,
            rnn_out=hidden_dim,
            dim_out=hidden_dim,
            n_layers=n_layers,
            bidirectional=bidirectional,
        )

        self.emb_model_2 = EmbedAttenSeq(
            dim_seq_in=CONFIGS[self.params["disease"]]["num_pub_features"],
            dim_metadata=0,
            rnn_out=hidden_dim,
            dim_out=hidden_dim,
            n_layers=n_layers,
            bidirectional=bidirectional,
        )

        self.decoder = DecodeSeq(
            dim_seq_in=1,
            rnn_out=hidden_dim, # divides by 2 if bidirectional
            dim_out=out_layer_dim,
            n_layers=1,
            bidirectional=True,
        ) 

        out_layer_width = out_layer_dim
        self.out_layer =  [
            nn.Linear(
                in_features=out_layer_width, out_features=out_layer_width//2
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=out_layer_width//2, out_features=out_dim
            ),
        ]
        self.out_layer = nn.Sequential(*self.out_layer)

        self.out_layer2 =  [
            nn.Linear(
                in_features=out_layer_width, out_features=out_layer_width//2
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=out_layer_width//2, out_features=CONFIGS[self.params['disease']]["num_patch"]
            ),
        ]
        self.out_layer2 = nn.Sequential(*self.out_layer2)


        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
        self.out_layer.apply(init_weights)
        self.out_layer2.apply(init_weights)
        self.min_values = torch.tensor(MIN_VAL_PARAMS[scale_output],device=self.device)
        self.max_values = torch.tensor(MAX_VAL_PARAMS[scale_output],device=self.device)
        self.min_values_2 = torch.tensor(MIN_VAL_PARAMS_2[scale_output],device=self.device)
        self.max_values_2 = torch.tensor(MAX_VAL_PARAMS_2[scale_output],device=self.device)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, meta, x_2):
        x_embeds, encoder_hidden = self.emb_model_1.forward(x.transpose(1, 0), meta)
        x_embeds_2, encoder_hidden_2 = self.emb_model_2.forward(x_2.transpose(1, 0))
        # print(x_2)
        # input()
        # print(x_embeds.shape, x_embeds_2.shape)
        # print(encoder_hidden.shape, encoder_hidden_2.shape)
        # input()
        x_embeds = torch.cat([x_embeds, x_embeds_2], dim=0)
        encoder_hidden = torch.cat([encoder_hidden, encoder_hidden_2], dim=1)
        # create input that will tell the neural network which week it is predicting
        # thus, we have one element in the sequence per each week of R0
        time_seq = torch.arange(1,self.training_weeks+WEEKS_AHEAD+1).repeat(x_embeds.shape[0],1).unsqueeze(2)
        Hi_data = ((time_seq - time_seq.min())/(time_seq.max() - time_seq.min())).to(self.device)
        emb = self.decoder(Hi_data, encoder_hidden, x_embeds) 
        out = self.out_layer(emb)
        out = torch.mean(out, dim=0)
        out = self.min_values + (self.max_values-self.min_values)*self.sigmoid(out) # (175, 5)

        emb_mean = torch.mean(emb, dim=0)
        # print(emb_mean.shape)
        # emb_mean = torch.mean(emb_mean, dim=0)
        emb_mean = emb_mean[-1, :]
        
        out2 = self.out_layer2(emb_mean)
        # print(out2.shape)
        # input()
        out2 = self.min_values_2 + (self.max_values_2-self.min_values_2)*self.sigmoid(out2) # (5)

        return out, out2


class ParamModel(nn.Module):
    def __init__(self, metas_train_dim, X_train_dim, device, hidden_dim=50, n_layers=2,out_dim=1, scale_output='abm-covid', bidirectional=True, CUSTOM_INIT=True):
        super().__init__()

        self.device = device
        self.emb_model = EmbedAttenSeq(
            dim_seq_in=X_train_dim,
            dim_metadata=metas_train_dim,
            dim_out=hidden_dim,
            n_layers=n_layers,
            bidirectional=bidirectional,
        ) 

        self.layer1 = nn.Linear(in_features=hidden_dim, out_features=20)
        # used to bypass the RNN - we want to check what's happening with gradients
        self.layer_bypass = nn.Linear(in_features=metas_train_dim, out_features=20)
        self.meanfc = nn.Linear(in_features=20, out_features=out_dim, bias=True)
        self.min_values = torch.tensor(MIN_VAL_PARAMS[scale_output],device=self.device)
        self.max_values = torch.tensor(MAX_VAL_PARAMS[scale_output],device=self.device)
        self.sigmoid = nn.Sigmoid()
        if CUSTOM_INIT:
            self.meanfc.bias = torch.nn.Parameter(torch.tensor([1.0]))

    def forward(self, x, meta):
        x_embeds = self.emb_model.forward(x.transpose(1, 0), meta)
        # use embedding for predicting: i) R0 and ii) Cases {for support counties} [FOR LATER]        
        ro_feats = self.layer1(x_embeds)
        ro_feats = nn.ReLU()(ro_feats)
        out = self.meanfc(ro_feats)
        # else:
        ''' bound output '''
        out = self.min_values + (self.max_values-self.min_values)*self.sigmoid(out)
        
        return out

class LearnableParams(nn.Module):
    ''' doesn't use data signals '''
    def __init__(self, num_params, device, scale_output='abm-covid'):
        super().__init__()
        self.device = device
        self.learnable_params = nn.Parameter(torch.rand(num_params, device=self.device))
        self.min_values = torch.tensor(MIN_VAL_PARAMS[scale_output],device=self.device)
        self.max_values = torch.tensor(MAX_VAL_PARAMS[scale_output],device=self.device)
        self.sigmoid = nn.Sigmoid()

    def forward(self):
        out = self.learnable_params
        ''' bound output '''
        out = self.min_values + (self.max_values-self.min_values)*self.sigmoid(out)
        return out

def normal(x, mu, sigma_sq):
    a = (-1*(Variable(x)-mu).pow(2)/(2*sigma_sq)).exp()
    b = 1/(2*sigma_sq*pi.expand_as(sigma_sq)).sqrt()
    return a*b    

def save_model(model,file_name,disease,region,week):
    PATH = os.path.join(SAVE_MODEL_PATH,disease,region)
    print(PATH)
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    torch.save(model.state_dict(), PATH+'/' + file_name+' '+week + ".pth")

def load_model(model,file_name,disease,region,week,device):
    PATH = os.path.join(SAVE_MODEL_PATH,disease,region)
    print(file_name)
    model.load_state_dict(torch.load(PATH+'/' + file_name+' '+week + ".pth",map_location=device))
    return model
def normalize_columns(arr):
    """Normalizes each column of a 2D NumPy array."""
    return (arr - arr.min(axis=0)) / (arr.max(axis=0) - arr.min(axis=0))

def param_model_forward(param_model,params,x,meta):
    # get R0 from county network
    if params['model_name'].startswith('GradABM-time-varying'):
        action_value = param_model.forward(x, meta)  # time-varying
    elif params['model_name'] == 'ABM-expert':
        if params['disease'] == 'COVID':
            action_value = torch.tensor([2.5, 0.02, 0.5])  # from CDC, for COVID -- previous I0 was 0.01
        if params['disease'] == 'Flu':
            action_value = torch.tensor([1.3, 1.0])  # from CDC, for COVID
        action_value = action_value.repeat((meta.shape[0],1))
    elif 'ABM-pred-correction' in params['model_name']: # same as SEIRM-static, but get 
        action_value = param_model.forward() 
        if params['disease']=='COVID':
            # NOTE: to fix, beta/gamma is for SIR, maybe not the same for SEIRM
            beta = action_value[0]
            gamma = action_value[2]
            mu = action_value[3]  # mortality rate
            initial_infections_percentage = action_value[4]
            action_value = torch.stack([beta/(gamma+mu),mu,initial_infections_percentage])
        elif params['disease']=='Flu':
            beta = action_value[0]
            # D = action_value[:,1]
            D = 3.5
            initial_infections_percentage = action_value[1]
            action_value = torch.stack([beta*D,initial_infections_percentage])
        action_value = action_value.reshape(1,-1) # make sure it's 2d
        print('R0 ABM-pred-correction',action_value)
    elif 'GradABM-learnable-params' in params['model_name']:
        action_value = param_model.forward()
        action_value = action_value.repeat((meta.shape[0],1))
    elif 'meta' in params['model_name']:
        if 'bogota' in params["disease"]:
            import pandas as pd
            df = pd.read_csv("Data/Processed/GHT_smaller.csv")
            SMOOTH_WINDOW = 7
            df = df.loc[:,['cases', 'covid-19 en colombia', 'covid-19 hoy', 'covid-19 bogota']].values
            # print(df)
            df = normalize_columns(df)
            # print(df)
            # input()
            df = torch.unsqueeze(torch.tensor(moving_average(df.ravel(),SMOOTH_WINDOW).reshape(-1, CONFIGS[params["disease"]]["num_pub_features"]), dtype=torch.float32), dim=0)

            param_prediction, seed_prediction, adjustment_matrix = param_model.forward(x, meta, df)  # time-varying
            action_value = [param_prediction, seed_prediction, adjustment_matrix]
        else:
            param_prediction, seed_prediction, adjustment_matrix = param_model.forward(x, meta)  # time-varying
            action_value = [param_prediction, seed_prediction, adjustment_matrix]
    else:
        raise ValueError('model name not valid')
    
    return action_value

def build_param_model(params,metas_train_dim,X_train_dim,device,CUSTOM_INIT=True):

    # get param dimension for ODE
    if params['disease']=='COVID':
        ode_param_dim = 5
        abm_param_dim = 3
        scale_output_ode = 'seirm'
        scale_output_abm = 'abm-covid'
    elif params['disease']=='Flu':
        ode_param_dim = 2
        abm_param_dim = 2
        scale_output_ode = 'sirs'
        scale_output_abm = 'abm-flu'
    if params['model_name'] == 'meta':
        abm_param_dim = 8
        ode_param_dim = 4
        scale_output_ode = 'seirm'
        scale_output_abm = 'meta'


    training_weeks  = params['num_steps'] / 7  # only needed for time-varying 
    # print(training_weeks, params['num_steps'])
    # input()
    assert training_weeks == int(training_weeks)
    

    ''' call constructor of param model depending on the model we want to run'''
    if params['model_name'].startswith('GradABM-time-varying'):
        param_model = CalibNN(params, metas_train_dim, X_train_dim, device, training_weeks, out_dim=abm_param_dim,scale_output=scale_output_abm).to(device)
    elif params['model_name'] == 'ABM-expert':
        param_model = None
    elif params['model_name'] == 'meta':
        param_model = CalibNNTwoEncoderThreeOutputs(params, metas_train_dim, X_train_dim, device, training_weeks, out_dim=abm_param_dim,scale_output=scale_output_abm).to(device)
    elif 'ABM-pred-correction' in params['model_name']:
        # load the param model from ODE
        # NOTE: currently it uses only R0
        param_model = LearnableParams(ode_param_dim,device,scale_output_ode).to(device)
    elif 'GradABM-learnable-params' in params['model_name']:
        param_model = LearnableParams(abm_param_dim,device,scale_output_abm).to(device)
    else:
        raise ValueError('model name not valid')
    return param_model

def build_simulator(params,devices,counties,seed_infection_status):
    ''' build simulator: ABM or ODE'''

    if 'ABM' in params['model_name']:
        if params['joint']:
            abm = {}
            # abm devices are different from the ones for the params model
            if len(devices) > 1:
                abm_devices = devices[1:]
            else:
                abm_devices = devices
            num_counties = len(counties)
            for c in range(num_counties):
                c_params = copy(params)
                c_params['county_id'] = counties[c]
                abm[counties[c]] = GradABM(c_params, abm_devices[c%len(abm_devices)])
        else:
            if len(devices) > 1:
                abm_device = devices[1]
            else:
                abm_device = devices[0]
            abm = GradABM(params, abm_device)

    elif 'ODE' in params['model_name']:
        if params['disease']=='COVID':
            abm = SEIRM(params, devices[0])
        elif params['disease']=='Flu':
            abm = SIRS(params, devices[0])
    elif 'meta' in params['model_name']:
        import pandas as pd
        root_directory = "../PETS_competition"
        usa_data = pd.read_excel(os.path.join(root_directory, "geo_data/county_usa.xlsx"))
        usa_data = usa_data[(usa_data["NAME_1"] == "Massachusetts") & (usa_data["NAME_2"] != "Dukes") & (usa_data["NAME_2"] != "Nantucket")]
        usa_flow_data = pd.read_csv(os.path.join(root_directory, "geo_data/USA_admin2_radiation_constant_0.01.csv"), sep=",")
        ma_flow_data = usa_flow_data.loc[usa_flow_data["origin"].str.startswith("USA.22.") & usa_flow_data["destination"].str.startswith("USA.22.") &
                                 (usa_flow_data["origin"] != "USA.22.4_1") &
                                 (usa_flow_data["destination"] != "USA.22.4_1") &
                                 (usa_flow_data["origin"] != "USA.22.10_1") &
                                 (usa_flow_data["destination"] != "USA.22.10_1") &
                                (usa_flow_data["origin"] != "USA.22.13_1") & 
                                (usa_flow_data["destination"] != "USA.22.13_1") & 
                                (usa_flow_data["origin"] != "USA.22.9_1") &
                                (usa_flow_data["destination"] != "USA.22.9_1")]

        ma_flow_data = ma_flow_data.pivot(index="origin", columns="destination", values="flow")
        ma_flow_data = ma_flow_data.fillna(200)
        ma_flow_data = ma_flow_data.apply(lambda x: x/sum(x), axis=1)
        

        population_data = pd.read_csv(os.path.join(root_directory, "./geo_data/USA_admin2_population.patchsim"), sep=" ", header=None)
        population_data = population_data[population_data[0].str.startswith("USA.22") & (population_data[0] != "USA.22.4_1") & (population_data[0] != "USA.22.9_1") & (population_data[0] != "USA.22.10_1") & (population_data[0] != "USA.22.13_1")]
        population_data = population_data.rename(columns={0:"state", 1:"population"})
        population_data.index = population_data["state"]

        abm = MetapopulationSEIRM(params, devices[0], 10, torch.Tensor(ma_flow_data.values), torch.Tensor(population_data['population'].values))
    
    if 'bogota' in params['disease']:
        import pandas as pd
        population_data = pd.read_csv("Data/Processed/population_bogota.csv")
        flow_data = pd.read_csv("Data/Processed/contact_matrix_bogota.csv", index_col=0)
        # print(flow_data.values.shape)
        # print(population_data['Population'].values.shape)
        print(123)
        abm = MetapopulationSEIRM(params, devices[0], CONFIGS[params['disease']]["num_patch"], torch.Tensor(flow_data.values), torch.Tensor(population_data['Population'].values))

    return abm

def forward_simulator(params,param_values,abm,training_num_steps,counties,devices):
    ''' assumes abm contains only one simulator for covid (one county), and multiple for flu (multiple counties)'''

    if params['joint']:
        num_counties = len(counties)
        predictions = torch.empty((num_counties,training_num_steps)).to(devices[0])

        if params["model_name"] == "meta":
            # for i in range(training_num_steps - 1):
            #     print(i)
            #     abm.single_step(i, param_values, seed_state='USA.22.5_1', I=10)
            #     # input()
            # for state_idx, state in enumerate(abm.models):
            #     daily = abm.models[state]["D"].values
            #     predictions[state_idx, :] = torch.Tensor(daily)
            #     input()
            for time_step in range(training_num_steps):
                # print(param_values.shape)
                params_epi, seed_status, adjustment_matrix = param_values[0], param_values[1], param_values[2]
                param_t = params_epi[time_step//7,:]
                _, pred_t = abm.step(time_step, param_t, seed_status, adjustment_matrix)
                predictions[:, time_step] = pred_t

        else:       
            for time_step in range(training_num_steps):
                if 'time-varying' in params['model_name']:
                    # print(param_values.shape)
                    # input()
                    param_t = param_values[:,time_step//7,:]
                    # print(training_num_steps)
                    # input("123123")
                else:
                    param_t = param_values
                # go over each abm
                for c in range(num_counties):
                    model_device = abm[counties[c]].device
                    population = abm[counties[c]].num_agents
                    _, pred_t = abm[counties[c]].step(time_step, param_t[c].to(model_device))
                    predictions[c,time_step] = pred_t.to(devices[0]) 
    else:
        num_counties = 1
        param_values = param_values.squeeze(0)
        predictions = []
        
        for time_step in range(training_num_steps):
            if 'time-varying' in params['model_name']:
                param_t = param_values[time_step//7,:]
            else:
                param_t = param_values
            model_device = abm.device
            _, pred_t = abm.step(time_step, param_t.to(model_device))
            predictions.append(pred_t.to(devices[0]))
        predictions = torch.stack(predictions,0).reshape(1,-1)  # num counties, seq len
        
    # post-process predictions for flu
    # targets are weekly, so we have to convert from daily to weekly
    if params['disease']=='Flu':
        predictions = predictions.reshape(num_counties,-1,7).sum(2)
    else:
        predictions = predictions.reshape(num_counties,-1)

    return predictions.unsqueeze(2)

def runner(params, devices, verbose, args, seed_infection_status):
    for run_id in range(params['num_runs']):
        print("Run: ", run_id)

        # set batch size depending on the number of devices
        # batch_size = max(len(devices)-1,1)
        batch_size = CONFIGS[params['disease']]["num_patch"]

        # get data loaders and ground truth targets
        if params['disease']=='COVID':
            if params['joint']:
                train_loader, metas_train_dim, X_train_dim, seqlen = \
                    fetch_county_data_covid(params['state'],'all',pred_week=params['pred_week'],batch_size=batch_size,noise_level=params['noise_level'], args=args)
                
                test_loader, _, _, _ = \
                    fetch_county_data_covid(params['state'],'all',pred_week=str(eval(params['pred_week']) + 4),batch_size=batch_size,noise_level=params['noise_level'], args=args)
                    
            else:
                train_loader, metas_train_dim, X_train_dim, seqlen = \
                    fetch_county_data_covid(params['state'],params['county_id'],pred_week=params['pred_week'],batch_size=batch_size,noise_level=params['noise_level'])
            params['num_steps'] = seqlen
        elif params['disease']=='Flu':
            if params['joint']:
                train_loader, metas_train_dim, X_train_dim, seqlen = \
                    fetch_county_data_flu(params['state'],'all',pred_week=params['pred_week'],batch_size=batch_size,noise_level=params['noise_level'])
            else:
                train_loader, metas_train_dim, X_train_dim, seqlen = \
                    fetch_county_data_flu(params['state'],params['county_id'],pred_week=params['pred_week'],batch_size=batch_size,noise_level=params['noise_level'])
            params['num_steps'] = seqlen * 7
        
        elif params['disease']=='bogota':
            train_loader, metas_train_dim, X_train_dim, seqlen = \
                fetch_age_group_data_covid(batch_size=batch_size,noise_level=params['noise_level'], args=args, split="train")
            
            test_loader, _, _, _ = \
                fetch_age_group_data_covid(batch_size=batch_size,noise_level=params['noise_level'], args=args, split="test")
            params['num_steps'] = seqlen

        # add days ahead to num steps because num steps is used for forward pass of param model
        training_num_steps = params['num_steps']
        params['num_steps'] += DAYS_HEAD 
        param_model = build_param_model(params,metas_train_dim,X_train_dim,devices[0],CUSTOM_INIT=True)
        # filename to save/load model
        file_name = 'param_model'+'_'+params['model_name']
        # do not train ABM because it uses a different calibration procedure
        train_flag = False if params['model_name'].startswith('ABM') or params['inference_only'] else True

        num_epochs = NUM_EPOCHS_DIFF
        CLIP = 10
        if 'learnable-params' in params['model_name']:
            lr = 1e-2  # obtained after tuning
            num_epochs *= 2
        elif 'meta' in params["model_name"]:
            lr = 1e-4  # obtained after tuning
        else:
            lr = 1e-4 if params['model_name'].startswith('GradABM') else 1e-4

        print(num_epochs)
        ''' step 1: training  ''' 
        if train_flag:
            param_model.train()
            assert param_model != None
            opt = torch.optim.Adam(filter(lambda p: p.requires_grad, param_model.parameters()),lr=CONFIGS[params["disease"]]["learning_rate"],weight_decay=0.01)
            # opt = DPOptimizer(opt,
            #         noise_multiplier=1.0,
            #         max_grad_norm=1.0,
            #         expected_batch_size=4,)

            loss_fcn = torch.nn.MSELoss(reduction='none')
            best_loss = np.inf
            losses = []
            for epi in range(num_epochs):
                start = time.time()
                batch_predictions = []
                if verbose:
                    print('\n',"="*60)
                    print("Epoch: ", epi)
                epoch_loss = 0
                for batch, (counties, meta, x, y) in enumerate(train_loader):   
                    # construct abm for each forward pass
                    # print(counties, x.shape, y.shape)
                    # input()
                    abm = build_simulator(copy(params),devices,counties, seed_infection_status)
                    # forward pass param model
                    meta = meta.to(devices[0])
                    x = x.to(devices[0])
                    y = y.to(devices[0])
                    param_values = param_model_forward(param_model,params,x,meta)
                    if verbose:
                        if param_values.dim()>2:
                            print(param_values[:,[0,-1],:])
                        else:
                            print(param_values)
                    # forward simulator for several time steps
                    if BENCHMARK_TRAIN:
                        start_bench = time.time()
                    predictions = forward_simulator(params,param_values,abm,training_num_steps,counties,devices)
                    # print(predictions[0, -10:])
                    # print(y[0, -10:])
                    if BENCHMARK_TRAIN:
                        # quit after 1 epoch
                        print('No steps:', training_num_steps)
                        print('time (s): ', time.time() - start_bench)
                        quit()
                    # loss
                    if verbose:
                        print(torch.cat((y,predictions),2))
                    loss_weight = torch.ones((len(counties),training_num_steps,1)).to(devices[0])
                    loss = (loss_weight*loss_fcn(y, predictions)).mean()
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(param_model.parameters(), CLIP)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    epoch_loss += torch.sqrt(loss.detach()).item()
                losses.append(epoch_loss/(batch+1))  # divide by number of batches
                print('epoch_loss',epoch_loss)

                if torch.isnan(loss):
                    break
                ''' save best model '''
                if epoch_loss < best_loss:
                    # print(predictions[0])
                    # print(param_values)
                    if params['joint']:
                        # for name, i in param_model.named_parameters():
                        #     print(i)
                        #     break
                        save_model(param_model,file_name,params['disease'],'joint',params['pred_week'])
                    else:
                        save_model(param_model,file_name,params['disease'],params['county_id'],params['pred_week'])
                    best_loss = epoch_loss
                
                print('epoch {} time (s): {:.2f}'.format(epi,time.time()- start))
    



        ''' step 2: inference step  ''' 
        ''' upload best model in inference ''' 
        param_model = None; abm = None
        
        param_model = build_param_model(copy(params),metas_train_dim, X_train_dim,devices[0],CUSTOM_INIT=True)
        param_model.eval()
        
        if not params['model_name'].startswith('ABM'):
            # load param model if it is not ABM-expert
            if params['joint']:
                param_model = load_model(param_model,file_name,params['disease'],'joint',params['pred_week'],devices[0])
            else:
                param_model = load_model(param_model,file_name,params['disease'],params['county_id'],params['pred_week'],devices[0])
        elif 'ABM-pred-correction' in params['model_name']:
            # pred-correction, uses param model from ODE
            file_name = 'param_model'+'_'+'DiffODE-learnable-params'
            if params['noise_level']>0:
                file_name = 'param_model'+'_'+'DiffODE-learnable-params'+'-noise' + str(params['noise_level'])
            param_model = load_model(param_model,file_name,params['disease'],params['county_id'],params['pred_week'],devices[0])

        # for name, i in param_model.named_parameters():
        #     print(i)
        #     break

        num_step = training_num_steps + DAYS_HEAD
        batch_predictions = []
        counties_predicted = []
        learned_params = []
        with torch.no_grad():
            for batch, (counties, meta, x, y) in enumerate(test_loader):
                # construct abm for each forward pass
                abm = build_simulator(params,devices,counties, seed_infection_status)
                # forward pass param model
                meta = meta.to(devices[0])
                x = x.to(devices[0])[:, :CONFIGS[params["disease"]]["train_days"], :]
                if 'meta' in params["model_name"]:
                    param_values = param_model_forward(param_model,params,x,meta)
                else:
                    param_values = param_model_forward(param_model,params,x,meta)
                # forward simulator for several time steps
                preds = forward_simulator(params,param_values,abm,num_step,counties,devices)
                # print(preds[0])
                batch_predictions.append(preds)
                counties_predicted.extend(counties)
                if 'meta' in params["model_name"]:
                    learned_params.extend(np.array(param_values[0].cpu().detach()))
                    learned_params.extend(np.array([param_values[1].cpu().detach()]))
                else:
                    learned_params.extend(np.array(param_values.cpu().detach()))
                
        # for item
        predictions = torch.cat(batch_predictions,axis=0)
        all_rmses = []
        mape_all = []
        all_mae = []

        for state_idx, target_values in enumerate(y):
            target = torch.squeeze(target_values.detach()).numpy()
            predictions_train = torch.squeeze(predictions.detach()[state_idx]).numpy()
            fig = plt.figure()
            plt.plot(target)
            plt.plot(predictions_train)
            rmse = np.sqrt(
            np.mean((target[:CONFIGS[params["disease"]]["train_days"]] -
                     predictions_train[:CONFIGS[params["disease"]]["train_days"]])**2))
            rmse_test = np.sqrt(
            np.mean((target[CONFIGS[params["disease"]]["train_days"]:(CONFIGS[params["disease"]]["train_days"]+CONFIGS[params["disease"]]["test_days"])] -
                     predictions_train[CONFIGS[params["disease"]]["train_days"]:(CONFIGS[params["disease"]]["train_days"]+CONFIGS[params["disease"]]["test_days"])])**2))
            mae_train = np.mean(np.absolute((target[:CONFIGS[params["disease"]]["train_days"]] -
                     predictions_train[:CONFIGS[params["disease"]]["train_days"]])))
            
            mae_test = np.mean(np.absolute(target[CONFIGS[params["disease"]]["train_days"]:(CONFIGS[params["disease"]]["train_days"]+CONFIGS[params["disease"]]["test_days"])] -
                     predictions_train[CONFIGS[params["disease"]]["train_days"]:(CONFIGS[params["disease"]]["train_days"]+CONFIGS[params["disease"]]["test_days"])]))
            mape_all.append(np.mean(np.abs((target[CONFIGS[params["disease"]]["train_days"]:(CONFIGS[params["disease"]]["train_days"]+CONFIGS[params["disease"]]["test_days"])] -
                     predictions_train[CONFIGS[params["disease"]]["train_days"]:(CONFIGS[params["disease"]]["train_days"]+CONFIGS[params["disease"]]["test_days"])])) / target[CONFIGS[params["disease"]]["train_days"]:(CONFIGS[params["disease"]]["train_days"]+CONFIGS[params["disease"]]["test_days"])]) * 100)
            plt.title('Training RMSE: {:.2f} Testing RMSE: {:.2f} Testing MAE {:.2f} '.format(rmse, rmse_test, mae_test))
            plt.xlabel("TimeStamp")
            plt.ylabel("Mortality Number")
            plt.legend(["Ground-truth", "Predictions"])
            fig.savefig(f"State_{state_idx}.png")
            all_rmses.append(rmse_test)
            all_mae.append(mae_test)
        print(sum(all_rmses)/len(all_rmses))
        print(sum(all_mae)/len(all_mae))
        print(sum(mape_all)/len(mape_all))
        

        # we only care about the last predictions
        # predictions are weekly, so we only care about the last 4
        if params['disease']=='Flu':
            predictions = predictions.squeeze(2)[:,-DAYS_HEAD//7:] 
        else: 
            predictions = predictions.squeeze(2)[:,-DAYS_HEAD:] 
        ''' remove grad '''
        predictions = predictions.cpu().detach()

        ''' release memory '''
        param_model = None; abm = None
        torch.cuda.empty_cache()

        ''' plot losses '''
        # only if trained
        if train_flag:
            disease = params['disease']
            if params['joint']:
                FIGPATH = f'./Figures/{disease}/joint/'
            else:
                county_id = params['county_id']
                FIGPATH = f'./Figures/{disease}/{county_id}/'
            if not os.path.exists(FIGPATH):
                os.makedirs(FIGPATH)
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax.plot(losses)
            # ax.set_ylim(2.5, 6.5)
            pred_week = params['pred_week']
            fig.savefig(FIGPATH+f'/losses_{pred_week}_{args.privacy}_{seed_infection_status}.png')


            

        print("-"*60)
        return counties_predicted, np.array(predictions), learned_params

def train_predict(args, seed_infection_status):

    # Setting seed
    print('='*60)
    if args.joint:
        print(f'state {args.state} week {args.pred_week}')
    else:
        print(f'county {args.county_id} week {args.pred_week}')
    print('Seed used for python random, numpy and torch is {}'.format(args.seed))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    params = {}            
    params['seed'] = args.seed
    params['num_runs'] = args.num_runs
    params['disease'] = args.disease
    params['pred_week'] = args.pred_week
    params['joint'] = args.joint
    params['inference_only'] = args.inference_only
    params['noise_level'] = args.noise  # for robustness experiments
    # state
    params['state'] = args.state
    if params['joint']:
        # verify it is a state
        assert params['state'] in states
    else:
        params['county_id'] = args.county_id
        # verify county belong to state
        assert params['county_id'] in counties[params['state']]
    params['model_name'] = args.model_name

    if args.dev == ['cpu']:
        devices = [torch.device("cpu")]
    else:
        devices = [torch.device(f'cuda:{i}') for i in args.dev]
    
    print('devices used:',devices)
    verbose = False
    
    global MIN_VAL_PARAMS, MAX_VAL_PARAMS, MIN_VAL_PARAMS_2, MAX_VAL_PARAMS_2
    MIN_VAL_PARAMS = {
        'abm-covid':[1.0, 0.001, 0.01],  # r0, mortality rate, initial_infections_percentage
        'abm-flu':[1.05, 0.1], # r0, initial_infections_percentage
        'seirm':[0., 0., 0., 0., 0.01], # beta, alpha, gamma, mu, initial_infections_percentage
        'sirs':[0., 0.1],  # beta, initial_infections_percentage
        'meta': [0, 0, 0, 0, 0, 0, 0, 0]
        }
    MAX_VAL_PARAMS = {
        'abm-covid':[8.0, 0.02, 1.0],  
        'abm-flu':[2.6, 5.0], 
        'seirm':[1., 1., 1., 1., 1.],
        'sirs':[1., 5.0], 
        'meta': [1, 1, 1, 1, 1, 1, 1, 1]
        # 'meta': [0.5, 0.3, 0.5, 0.5, 0.3, 0.8, 0.5, 1]
        }


    MIN_VAL_PARAMS_2 = {
        'abm-covid':[1.0, 0.001, 0.01],  # r0, mortality rate, initial_infections_percentage
        'abm-flu':[1.05, 0.1], # r0, initial_infections_percentage
        'seirm':[0., 0., 0., 0., 0.01], # beta, alpha, gamma, mu, initial_infections_percentage
        'sirs':[0., 0.1],  # beta, initial_infections_percentage
        'meta': [0] *  (CONFIGS[params['disease']]["num_patch"]) 
        # 'meta': [0] * 4 + [10] * (CONFIGS[params['disease']]["num_patch"] - 4)
        }
        
    MAX_VAL_PARAMS_2 = {
        'abm-covid':[8.0, 0.02, 1.0],  
        'abm-flu':[2.6, 5.0], 
        'seirm':[1., 1., 1., 1., 1.],
        'sirs':[1., 5.0], 
        # 'meta': [0] * 4 + [20] * (CONFIGS[params['disease']]["num_patch"] - 4)
        'meta': [10] * (CONFIGS[params['disease']]["num_patch"])
        }
    
    global transaction_data, meta2
    # load the pre-processed transaction dataset and normalize the dataset
    transaction_data = torch.load("./Data/Processed/online/transaction_private_lap_0_moving.pt").to(torch.float32).unsqueeze(2)
    transaction_data = torch.nn.functional.normalize(transaction_data,dim=0).to(devices[0])
    meta2 = torch.eye(transaction_data.shape[0]).to(devices[0])

    counties_predicted, predictions, learned_params = runner(params, devices, verbose, args, seed_infection_status=seed_infection_status)

    return counties_predicted, predictions, learned_params