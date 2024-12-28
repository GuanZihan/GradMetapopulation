import random
import numpy as np
import torch
import os
import torch.nn as nn
import time
from copy import copy
import pandas as pd
from torch.autograd import Variable

from data_utils import WEEKS_AHEAD, states, counties
from model_utils_quantile import EmbedAttenSeq, fetch_county_data_covid, fetch_county_data_flu, DecodeSeq, SEIRM, SIRS, MetapopulationSEIRM, fetch_age_group_data_covid, moving_average, MetapopulationSEIRMBeta
from visualize_results import *


class MultiQuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super(MultiQuantileLoss, self).__init__()
        self.quantiles = quantiles

    def forward(self, y_pred, y_true):
        total_loss = 0.0
        for idx, quantile in enumerate(self.quantiles):
            # print(len(y_true), len(y_pred[idx]))
            # Compute the quantile-specific error
            error = y_true - y_pred[idx]
            quantile_loss = torch.max(quantile * error, (quantile - 1) * error)
            total_loss += torch.mean(quantile_loss)
        
        # Return the average loss across all quantiles
        return total_loss / len(self.quantiles)


# define the moving average function
def moving_average(x, w):
    return pd.Series(x).rolling(w, min_periods=1).mean().values

# define the dataset class for the public dataset (Time Series)
class SeqDataset(torch.utils.data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, target_name):
        self.data = pd.read_csv(csv_file).values
        self.targets = moving_average(pd.read_csv(csv_file)[target_name].values.ravel()[:CONFIGS['bogota']["train_days"]],SMOOTH_WINDOW).reshape(-1,1)
        self.targets = np.concatenate([self.targets, pd.read_csv(csv_file)[target_name].values.ravel()[CONFIGS['bogota']["train_days"]:].reshape(-1,1)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = torch.from_numpy(self.data[idx].astype(np.float32))
        y = self.targets[idx].astype(np.float32)
        return X, y

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

        out3 = self.sigmoid(self.out_layer3(emb_mean).reshape((CONFIGS[self.params['disease']]["num_patch"], CONFIGS[self.params['disease']]["num_patch"])))

        return out, out2, out3

# define the neural network fore predicting epi-parameters
class CalibNNTwoEncoderThreeOutputs(nn.Module):
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

        self.emb_model_2 = EmbedAttenSeq(
            dim_seq_in=CONFIGS[self.params["disease"]]["num_pub_features"],
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
                in_features=out_layer_width//2, out_features=out_dim*3
            ),
        ]
        self.out_layer = nn.Sequential(*self.out_layer)

        self.out_layer2 =  [
            nn.Linear(
                in_features=out_layer_width, out_features=out_layer_width//2
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=out_layer_width//2, out_features=CONFIGS[self.params['disease']]["num_patch"]*3
            ),
        ]
        self.out_layer2 = nn.Sequential(*self.out_layer2)

        self.out_layer3 =  [
            nn.Linear(
                in_features=out_layer_width, out_features=out_layer_width//2
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=out_layer_width//2, out_features=CONFIGS[self.params['disease']]["num_patch"]*CONFIGS[self.params['disease']]["num_patch"]*3
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

    def forward(self, x, meta, x_2, meta2):
        # emb_model: handling zipcode-level private transaction dataset
        x_embeds, encoder_hidden = self.emb_model.forward(x.transpose(1, 0), meta)
        # emb_model_2: handling city-level public dataset
        x_embeds_2, encoder_hidden_2 = self.emb_model_2.forward(x_2.transpose(1, 0), meta2)
        # concatenate embeddings
        x_embeds = torch.cat([x_embeds, x_embeds_2.mean(dim=0).unsqueeze(0)], dim=0)
        encoder_hidden = torch.cat([encoder_hidden, encoder_hidden_2.mean(dim=1).unsqueeze(1)], dim=1)

        time_seq = torch.arange(1,self.training_weeks+WEEKS_AHEAD+1).repeat(x_embeds.shape[0],1).unsqueeze(2)
        Hi_data = ((time_seq - time_seq.min())/(time_seq.max() - time_seq.min())).to(self.device)
        emb = self.decoder(Hi_data, encoder_hidden, x_embeds) 
        # print(x.shape, emb.shape)
        out = self.out_layer(emb)
        out = torch.mean(out, dim=0)
        out = out.reshape(-1, 7, 3)
        # `out` contains the predicted epi parameters except for beta
        out = self.min_values.reshape(1, 7, 1) + (self.max_values.reshape(1, 7, 1)-self.min_values.reshape(1, 7, 1))*self.sigmoid(out)

        emb_mean = torch.mean(emb, dim=0)
        emb_mean = emb_mean[-1, :]
        
        out2 = self.out_layer2(emb_mean).reshape(-1, 16, 3)
        # `out` contains the predicted `seed_status`
        out2 = self.min_values_2.reshape(1, 16, 1) + (self.max_values_2.reshape(1, 16, 1)-self.min_values_2.reshape(1, 16, 1))*self.sigmoid(out2) # (5)

        # `out3` contains the predicted beta matrix
        # print(self.out_layer3(emb_mean).shape)
        out3 = self.sigmoid(self.out_layer3(emb_mean).reshape((CONFIGS[self.params['disease']]["num_patch"], CONFIGS[self.params['disease']]["num_patch"], 3)))
        
        return out, out2, out3


# define the neural network fore predicting epi-parameters
class CalibNNThreeEncoderThreeOutputs(nn.Module):
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

        self.emb_model_2 = EmbedAttenSeq(
            dim_seq_in=CONFIGS[self.params["disease"]]["num_pub_features"],
            dim_metadata=80,
            rnn_out=hidden_dim,
            dim_out=hidden_dim,
            n_layers=n_layers,
            bidirectional=bidirectional,
        )

        self.emb_model_3 = EmbedAttenSeq(
            dim_seq_in=6,
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

    def forward(self, x, meta, x_2, meta2):
        # emb_model: handling zipcode-level private transaction dataset
        x_embeds, encoder_hidden = self.emb_model.forward(x[:, :, :5].transpose(1, 0), meta)
        # emb_model_2: handling city-level public dataset
        x_embeds_2, encoder_hidden_2 = self.emb_model_2.forward(x_2.transpose(1, 0), meta2)
        x_embeds_3, encoder_hidden_3 = self.emb_model_3.forward(x[:, :, 5:].transpose(1, 0), meta)
        # concatenate embeddings
        x_embeds = torch.cat([x_embeds, x_embeds_3, x_embeds_2.mean(dim=0).unsqueeze(0)], dim=0)
        encoder_hidden = torch.cat([encoder_hidden, encoder_hidden_3, encoder_hidden_2.mean(dim=1).unsqueeze(1)], dim=1)

        time_seq = torch.arange(1,self.training_weeks+WEEKS_AHEAD+1).repeat(x_embeds.shape[0],1).unsqueeze(2)
        Hi_data = ((time_seq - time_seq.min())/(time_seq.max() - time_seq.min())).to(self.device)
        emb = self.decoder(Hi_data, encoder_hidden, x_embeds) 
        out = self.out_layer(emb)
        out = torch.mean(out, dim=0)
        # `out` contains the predicted epi parameters except for beta
        out = self.min_values + (self.max_values-self.min_values)*self.sigmoid(out)

        emb_mean = torch.mean(emb, dim=0)
        emb_mean = emb_mean[-1, :]
        
        out2 = self.out_layer2(emb_mean)
        # `out` contains the predicted `seed_status`
        out2 = self.min_values_2 + (self.max_values_2-self.min_values_2)*self.sigmoid(out2) # (5)

        # `out3` contains the predicted beta matrix
        out3 = self.sigmoid(self.out_layer3(emb_mean).reshape((CONFIGS[self.params['disease']]["num_patch"], CONFIGS[self.params['disease']]["num_patch"])))
        
        return out, out2, out3

def save_model(model,file_name,disease,region,week, args):
    PATH = os.path.join(SAVE_MODEL_PATH,disease,region, args.date)
    print(PATH)
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    torch.save(model.state_dict(), PATH+'/' + file_name+' '+week + ".pth")

def load_model(model,file_name,disease,region,week,device, args):
    PATH = os.path.join(SAVE_MODEL_PATH,disease,region, args.date)
    print(file_name)
    model.load_state_dict(torch.load(PATH+'/' + file_name+' '+week + ".pth",map_location=device))
    return model

def param_model_forward(param_model,params,x,meta):
    if 'meta' in params['model_name']:
        if 'bogota' in params["disease"]:
            # data and meta2 have been initialized before
            param_prediction, seed_prediction, adjustment_matrix = param_model.forward(x, meta, data, meta2)  # time-varying
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
    if params['model_name'] == 'meta':
        abm_param_dim = 7
        ode_param_dim = 4
        scale_output_ode = 'seirm'
        scale_output_abm = 'meta'
    training_weeks  = params['num_steps'] / 7  # only needed for time-varying 
    assert training_weeks == int(training_weeks)

    ''' call constructor of param model depending on the model we want to run'''
    if params['model_name'] == 'meta':
        param_model = CalibNNTwoEncoderThreeOutputs(params, metas_train_dim, X_train_dim, device, training_weeks, out_dim=abm_param_dim,scale_output=scale_output_abm).to(device)
        # param_model = CalibNNThreeOutputs(params, metas_train_dim, X_train_dim, device, training_weeks, out_dim=abm_param_dim,scale_output=scale_output_abm).to(device)

    return param_model

def build_simulator(params,devices):
    ''' 
    build simulator: MetaPopulation
    contact matrix and population data are extracted from the public source
    '''

    if 'meta' in params['model_name'] and 'COVID' in params["disease"]:
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

        abm = MetapopulationSEIRMBeta(params, devices[0], 10, torch.Tensor(ma_flow_data.values), torch.Tensor(population_data['population'].values))
    
    elif 'bogota' in params['disease']:
        population_data = pd.read_csv("Data/Processed/population_bogota.csv")
        flow_data = pd.read_csv("Data/Processed/contact_matrix_bogota.csv", index_col=0)
        abm = MetapopulationSEIRMBeta(params, devices[0], CONFIGS[params['disease']]["num_patch"], torch.Tensor(flow_data.values), torch.Tensor(population_data['Population'].values))

    return abm

def forward_simulator(params,param_values,abms,training_num_steps,counties,devices):
    if params['joint']:
        num_counties = len(counties)

        all_predictions = []
        
        if params["model_name"] == "meta":
            for idx, abm in enumerate(abms):
                predictions = torch.empty((num_counties,training_num_steps)).to(devices[0])
                for time_step in range(training_num_steps):
                    # split the predicted epi-parameters of the neural network
                    # print(param_values.shape)
                    params_epi, seed_status, adjustment_matrix = param_values[0][:, :, idx], param_values[1][0, :, idx], param_values[2][:, :, idx]
                    # print(params_epi.shape, seed_status.shape, adjustment_matrix.shape)
                    # choose the epi-parameter according to the time step
                    param_t = params_epi[time_step//7,:]
                    # go simulation step
                    _, pred_t = abm.step(time_step, param_t, seed_status, adjustment_matrix)
                    # save the prediction
                    predictions[:, time_step] = pred_t
                predictions = predictions.reshape(num_counties,-1)
                predictions = torch.sum(predictions, dim=0).unsqueeze(0)
                predictions = predictions.unsqueeze(2)
                all_predictions.append(predictions)

    
    # print(all_predictions)
    # print(all_predictions[0].shape)
    # input()
    
    return all_predictions

def runner(params, devices, verbose, args):
    for run_id in range(params['num_runs']):
        print("Run: ", run_id)

        # set batch size depending on the number of devices
        # Training with full-batch
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
        elif params['disease']=='bogota':
            train_dataset = SeqDataset("./Data/Processed/online/train_{}.csv".format(args.date), "cases")
            test_dataset = SeqDataset("./Data/Processed/online/test_{}.csv".format(args.date), "cases")

            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=800, shuffle=False)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=800, shuffle=False)

            metas_train_dim = 1
            X_train_dim = 11

            params['num_steps'] = len(train_dataset)


        # add days ahead to num steps because num steps is used for forward pass of param model
        training_num_steps = params['num_steps']
        params['num_steps'] += DAYS_HEAD 
        param_model = build_param_model(params,metas_train_dim,X_train_dim,devices[0],CUSTOM_INIT=True)
        # filename to save/load model
        file_name = 'param_model'+'_'+params['model_name']
        # set if train the meta-population model
        train_flag = False if params['inference_only'] else True

        num_epochs = NUM_EPOCHS_DIFF
        # gradient clipping norm
        CLIP = 10
        if 'learnable-params' in params['model_name']:
            lr = 1e-2  # obtained after tuning
            num_epochs *= 2
        elif 'meta' in params["model_name"]:
            lr = 1e-4  # obtained after tuning
        else:
            lr = 1e-4 if params['model_name'].startswith('GradABM') else 1e-4

        ''' step 1: training  ''' 
        if train_flag:
            param_model.train()
            param_model.to(devices[0])
            assert param_model != None
            opt = torch.optim.Adam(filter(lambda p: p.requires_grad, param_model.parameters()),lr=CONFIGS[params["disease"]]["learning_rate"],weight_decay=0.01)
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
                for batch, (x, y) in enumerate(train_loader):   
                    abm1 = build_simulator(copy(params),devices)
                    abm2 = build_simulator(copy(params),devices)
                    abm3 = build_simulator(copy(params),devices)
                    # forward pass param model
                    meta = torch.Tensor([[1]])
                    # we have 16 age groups
                    counties = torch.ones(16)
                    meta = meta.to(devices[0])
                    x = x.to(devices[0]).unsqueeze(0)
                    y = y.to(devices[0]).unsqueeze(0)
                    param_values = param_model_forward(param_model,params,x,meta)
                    if verbose:
                        if param_values.dim()>2:
                            print(param_values[:,[0,-1],:])
                        else:
                            print(param_values)
                    # forward simulator for several time steps
                    if BENCHMARK_TRAIN:
                        start_bench = time.time()
                    # get predictions after running meta-population model
                    predictions = forward_simulator(params,param_values,[abm1,abm2,abm3],training_num_steps,counties,devices)
                    if BENCHMARK_TRAIN:
                        # quit after 1 epoch
                        print('No steps:', training_num_steps)
                        print('time (s): ', time.time() - start_bench)
                        quit()
                    # loss
                    if verbose:
                        print(torch.cat((y,predictions),2))

                    # loss weight is set as all-one values
                    loss_weight = torch.ones((len(counties),training_num_steps,1)).to(devices[0])
                    quantiles = [0.2, 0.5, 0.8]
                    loss_fn = MultiQuantileLoss(quantiles)
                    loss = (loss_weight*loss_fn(predictions, y)).mean()
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(param_model.parameters(), CLIP)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    epoch_loss += torch.sqrt(loss.detach()).item()
                losses.append(epoch_loss/(batch+1))  # divide by number of batches
                print('epoch_loss',epoch_loss, batch)

                if torch.isnan(loss):
                    break
                ''' save best model '''
                if epoch_loss < best_loss:
                    if params['joint']:
                        save_model(param_model,file_name,params['disease'],'joint',params['pred_week'], args)
                    else:
                        save_model(param_model,file_name,params['disease'],params['county_id'],params['pred_week'], args)
                    best_loss = epoch_loss
                    print("current best loss is {}".format(best_loss))
                print('epoch {} time (s): {:.2f}'.format(epi,time.time()- start))

        ''' step 2: inference step  ''' 
        ''' upload best model in inference ''' 
        param_model = None; abm = None
        
        param_model = build_param_model(copy(params),metas_train_dim, X_train_dim,devices[0],CUSTOM_INIT=True)
        param_model.eval()
        
        
        # load param model from the saved directory
        if params['joint']:
            param_model = load_model(param_model,file_name,params['disease'],'joint',params['pred_week'],devices[0], args)
        else:
            param_model = load_model(param_model,file_name,params['disease'],params['county_id'],params['pred_week'],devices[0], args)

        num_step = training_num_steps + DAYS_HEAD # adding number of prediction horizon to the simulation step
        batch_predictions = []
        counties_predicted = []
        learned_params = []
        with torch.no_grad():
            for batch, (x, y) in enumerate(test_loader):
                abm1 = build_simulator(copy(params),devices)
                abm2 = build_simulator(copy(params),devices)
                abm3 = build_simulator(copy(params),devices)
                # forward pass param model
                meta = torch.Tensor([[1]])
                counties = torch.ones(16)
                meta = meta.to(devices[0])
                x = x.to(devices[0]).unsqueeze(0)[:, :CONFIGS[params["disease"]]["train_days"], :]
                if 'meta' in params["model_name"]:
                    param_values = param_model_forward(param_model,params,x,meta)
                else:
                    param_values = param_model_forward(param_model,params,x,meta)
                
                # forward simulator for several time steps
                
                preds = forward_simulator(params,param_values,[abm1,abm2,abm3],num_step,counties,devices)
                # print(preds)
                # print(preds.shape)
                batch_predictions.append(preds)
                counties_predicted.extend(counties)
                if 'meta' in params["model_name"]:
                    learned_params.extend(np.array(param_values[0].cpu().detach()))
                    learned_params.extend(np.array([param_values[1].cpu().detach()]))
                    learned_params.extend(np.array([param_values[2].cpu().detach()]))
                else:
                    learned_params.extend(np.array(param_values.cpu().detach()))

        predictions = torch.cat(batch_predictions[0],axis=0)
        

        # calculate training and testing RMSE values
        all_rmses = []
        all_maes = []
        all_mapes = []
        training_mae = []
        all_mapes_train = []
        for state_idx, target_values in enumerate(y.unsqueeze(0)):
            target = torch.squeeze(target_values.detach()).numpy()
            predictions_train = torch.squeeze(predictions.detach()[1]).cpu().numpy()
            # print(target[(CONFIGS[params["disease"]]["train_days"]-10):CONFIGS[params["disease"]]["train_days"]], predictions_train[(CONFIGS[params["disease"]]["train_days"]-10):CONFIGS[params["disease"]]["train_days"]])
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

            # plot_predictions(target, predictions[0,:,0], rmse, rmse_test, state_idx, args)
            # plot_predictions(target, predictions[1,:,0], rmse, rmse_test, state_idx, args)
            # plot_predictions(target, predictions[2,:,0], rmse, rmse_test, state_idx, args)

            fig = plt.figure()
            plt.plot(target)
            plt.plot(predictions[0,:,0])
            print(predictions.shape)
            plt.plot(predictions[1,:,0])
            plt.plot(predictions[2,:,0])
            plt.title('Training RMSE: {:.2f} Testing RMSE: {:.2f}'.format(rmse, rmse_test))
            plt.xlabel("TimeStamp")
            plt.ylabel("Mortality Number")
            plt.legend(["Ground-truth", "Predictions with quantile 0.2", "Predictions with quantile 0.5", "Predictions with quantile 0.8"])
            fig.savefig(os.path.join("Figure-Prediction", f"State_{state_idx}_{args.date}_{args.note}.png"))

            
            all_rmses.append(rmse_test)
            all_maes.append(mae_test)
            all_mapes.append(np.mean(np.abs((target[CONFIGS[params["disease"]]["train_days"]:(CONFIGS[params["disease"]]["train_days"]+CONFIGS[params["disease"]]["test_days"])] -
                     predictions_train[CONFIGS[params["disease"]]["train_days"]:(CONFIGS[params["disease"]]["train_days"]+CONFIGS[params["disease"]]["test_days"])])) / target[CONFIGS[params["disease"]]["train_days"]:(CONFIGS[params["disease"]]["train_days"]+CONFIGS[params["disease"]]["test_days"])]) * 100)
            all_mapes_train.append(np.mean(np.abs((target[:CONFIGS[params["disease"]]["train_days"]] -
                     predictions_train[:CONFIGS[params["disease"]]["train_days"]])) / target[:CONFIGS[params["disease"]]["train_days"]]) * 100)
            training_mae.append(mae_train)
        print('RMSE: ', sum(all_rmses)/len(all_rmses))
        print(all_maes)
        print('testing MAE: ', sum(all_maes)/len(all_maes))
        print('training MAE: ', sum(training_mae)/len(training_mae))
        print('training MAPES', sum(all_mapes_train)/len(all_mapes_train))
        print('testing MAPES: ', sum(all_mapes)/len(all_mapes))

        # we only care about the last predictions
        # predictions are weekly, so we only care about the last 4
        predictions = predictions.squeeze(2)[:,-DAYS_HEAD:] 
        ''' remove grad '''
        predictions = predictions.cpu().detach()

        ''' release memory '''
        param_model = None; abm = None
        torch.cuda.empty_cache()

        ''' plot losses '''
        # only if trained, plot training loss curve
        if train_flag:   
            # plot all the losses
            plot_losses(losses, params, args)
        print("-"*60)
        return counties_predicted, np.array(predictions), learned_params

def train_predict(args):
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

    global devices
    if args.dev == ['cpu']:
        devices = [torch.device("cpu")]
    else:
        devices = [torch.device(f'cuda:{i}') for i in args.dev]
    
    print('devices used:',devices)
    

    global data, meta2
    # load the pre-processed transaction dataset and normalize the dataset
    data = torch.load("./Data/Processed/online/transaction_private_lap_{}.pt".format(args.date)).to(torch.float32).unsqueeze(2)
    data = torch.nn.functional.normalize(data,dim=0).to(devices[0])
    meta2 = torch.eye(data.shape[0]).to(devices[0])

    global CONFIGS
    # configs for different datasets
    CONFIGS = {
        "bogota": {
            "num_patch": 16,
            "learning_rate": 5e-5,
            "train_days": 343 + eval(args.date.split("_")[0]) * 7,
            "test_days": 28,
            "num_pub_features": 1
        },
        "COVID": {
            "num_patch": 10,
            "learning_rate": 1e-4,
            "train_days": 175,
            "test_days": 28
        }
    }

    global BENCHMARK_TRAIN, NUM_EPOCHS_DIFF, DAYS_HEAD, SAVE_MODEL_PATH, SMOOTH_WINDOW
    # if calculating training time
    BENCHMARK_TRAIN = False
    # number of epochs
    NUM_EPOCHS_DIFF = 5000
    print("---- MAIN IMPORTS SUCCESSFUL -----")



    # define the prediction horizon
    DAYS_HEAD = 4*7  # 4 weeks ahead

    # define the directory for saving the model
    SAVE_MODEL_PATH = './Models/'

    # define the size of smooth window
    SMOOTH_WINDOW=7


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
    params['date'] = args.date

    
    verbose = False
    
    # upper and lower bounds of the neural network prediction
    global MIN_VAL_PARAMS, MAX_VAL_PARAMS, MIN_VAL_PARAMS_2, MAX_VAL_PARAMS_2
    MIN_VAL_PARAMS = {
        'abm-covid':[1.0, 0.001, 0.01],  # r0, mortality rate, initial_infections_percentage
        'abm-flu':[1.05, 0.1], # r0, initial_infections_percentage
        'seirm':[0., 0., 0., 0., 0.01], # beta, alpha, gamma, mu, initial_infections_percentage
        'sirs':[0., 0.1],  # beta, initial_infections_percentage
        'meta': [0, 0, 0, 0, 0, 0, 0]
        }
    MAX_VAL_PARAMS = {
        'abm-covid':[8.0, 0.02, 1.0],  
        'abm-flu':[2.6, 5.0], 
        'seirm':[1., 1., 1., 1., 1.],
        'sirs':[1., 5.0], 
        'meta': [1, 1, 1, 1, 1, 1, 1]
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
    
    # start training!
    counties_predicted, predictions, learned_params = runner(params, devices, verbose, args)

    return counties_predicted, predictions, learned_params
    