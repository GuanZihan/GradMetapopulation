import random
import numpy as np
import torch
import os
import torch.nn as nn
import time
from copy import copy, deepcopy
import pandas as pd
from torch.autograd import Variable

from data_utils import WEEKS_AHEAD, states, counties
from model_utils import EmbedAttenSeq, fetch_county_data_covid, DecodeSeq, MetapopulationSEIRMBeta
from visualize_results import *

from models_pets import LSTM_MCDO, LSTM_Two_Encoder
from utils import *
import yaml
import logging


import pdb

from tqdm import tqdm

# Configure module logger. If the root logger has no handlers, configure a sensible default.
logger = logging.getLogger(__name__)
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

logging.getLogger().setLevel(logging.INFO)

def save_model(model: nn.Module, file_name: str, disease: str, args) -> None:
    """Persist the model to disk under the configured save path.

    The path is structured as SAVE_MODEL_PATH/disease/date/note.
    """
    PATH = os.path.join(SAVE_MODEL_PATH, disease, args.date, args.note)
    os.makedirs(PATH, exist_ok=True)
    torch.save(model.state_dict(), PATH + '/' + file_name + ".pth")

def load_model(model: nn.Module, file_name: str, disease: str, device: torch.device, args) -> nn.Module:
    """Load model weights from disk and return the model instance."""
    PATH = os.path.join(SAVE_MODEL_PATH, disease, args.date, args.note)
    logger.info("Loading model checkpoint from %s", PATH)
    model.load_state_dict(torch.load(PATH + '/' + file_name + ".pth", map_location=device))
    return model

# define the dataset class for the public dataset (Time Series)
class SeqDataset(torch.utils.data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, target_name, gt_file, masked_feature=""):
        df = pd.read_csv(csv_file)
        print("<-------> ", df.shape)
        if masked_feature != "":
            df.drop(masked_feature, axis=1, inplace=True)
        print("<-------> ", df.shape)
        self.data = df.values
        self.targets = pd.read_csv(gt_file)[target_name].values.ravel().reshape(-1,1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = torch.from_numpy(self.data[idx].astype(np.float32))
        y = self.targets[idx].astype(np.float32)
        return X, y

# define the neural network fore predicting epi-parameters
class CalibNNOneEncoderThreeOutputs(nn.Module):
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

        self.n_back = 4
        self.n_ahead = 1
        self.input_shape = self.n_back
        self.hidden_rnn = 32
        self.hidden_dense = 16
        self.output_dim = self.n_ahead
        self.activation = 'relu'
        self.n_feature = 1
        self.n_in = self.n_back * self.n_feature
        self.n_out = self.n_ahead * self.n_feature
        
        self.lstm_model = nn.Sequential(
            nn.Linear(5,256),
            nn.ReLU(),
            nn.Linear(256,1)
        )

    def forward(self, x, meta, train_X, train_Y):
        # emb_model: handling zipcode-level private transaction dataset
        x_embeds, encoder_hidden = self.emb_model.forward(x.transpose(1, 0), meta)
        time_seq = torch.arange(1,self.training_weeks+WEEKS_AHEAD+1).repeat(x_embeds.shape[0],1).unsqueeze(2)
        Hi_data = ((time_seq - time_seq.min())/(time_seq.max() - time_seq.min())).to(self.device)
        # print(Hi_data.shape)
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
        # return out, out2, out3, outputs, train_Y
        return out, out2, out3, train_Y
    
    def forward_lstm(self, x, x_exo):
        x_ = x * scaler.scale_[0] + scaler.min_[0]
        # pdb.set_trace()
        train_X = x_.squeeze().unfold(0,5,1).unsqueeze(-1)[:,-5:,-1]
        outputs = self.lstm_model(train_X)
        return outputs

def param_model_forward(param_model,params,x,meta):
    if 'meta' in params['model_name']:
        # data and meta2 have been initialized before
        x_ = scaler.transform(x.cpu()[:, :, 0].T)
        reframed = series_to_supervised(x_, 4, 1)
        reframed_data = torch.Tensor(reframed.values)
        reframed_data = torch.unsqueeze(reframed_data, 1)
        train_X, train_Y = reframed_data[:, :, :4].to(devices[0]).permute(0, 2, 1), reframed_data[:, :, -1:].to(devices[0])
        param_prediction, seed_prediction, adjustment_matrix, targets = param_model.forward(x, meta, train_X, train_Y)  # time-varying
        # print(x.shape)
        action_value = [param_prediction, seed_prediction, adjustment_matrix]
    else:
        raise ValueError('model name not valid')
    
    # pdb.set_trace()
    
    # return action_value, prediction, targets
    return action_value, targets

def build_param_model(params,metas_train_dim,X_train_dim,device,CUSTOM_INIT=True):
    # get param dimension for ODE
    if params['disease']=='COVID':
        abm_param_dim = 3
        scale_output_ode = 'seirm'
        scale_output_abm = 'abm-covid'
    if params['model_name'] == 'meta':
        abm_param_dim = 7
        scale_output_ode = 'seirm'
        scale_output_abm = 'meta'
    training_weeks  = params['num_steps'] # only needed for time-varying 

    ''' call constructor of param model depending on the model we want to run'''
    if params['model_name'] == 'meta':
        param_model = CalibNNOneEncoderThreeOutputs(params, metas_train_dim, X_train_dim, device, training_weeks, out_dim=abm_param_dim, scale_output=scale_output_abm).to(device)
    return param_model

def build_simulator(params,devices):
    ''' 
    build simulator: MetaPopulation
    contact matrix and population data are extracted from the public source
    '''

    population_data = pd.read_csv("Data/US/population_us.csv")
    flow_data = pd.read_csv("Data/US/contact_matrix_us.csv", index_col=0)
    abm = MetapopulationSEIRMBeta(params, devices[0], CONFIGS[params['disease']]["num_patch"], torch.Tensor(flow_data.values), torch.Tensor(population_data['Population'].values))
    return abm

def forward_simulator(params,param_values,abm,training_num_steps,counties,devices):
    num_counties = len(counties)
    predictions = torch.empty((num_counties,training_num_steps)).to(devices[0])
    if params["model_name"] == "meta":
        for time_step in range(training_num_steps):
            # split the predicted epi-parameters of the neural network
            params_epi, seed_status, adjustment_matrix = param_values[0], param_values[1], param_values[2]
            # choose the epi-parameter according to the time step
            param_t = params_epi[time_step//7,:]
            # go simulation step
            _, pred_t = abm.step(time_step, param_t, seed_status, adjustment_matrix)
            # save the prediction
            predictions[:, time_step] = pred_t
        
    predictions = predictions.reshape(num_counties,-1)
    predictions = torch.sum(predictions, dim=0).unsqueeze(0)
    return predictions.unsqueeze(2)

def runner(params, devices, verbose, args):
    for run_id in range(params['num_runs']):
        # set batch size depending on the number of devices
        # Training with full-batch
        batch_size = CONFIGS[params['disease']]["num_patch"]
            
        train_datasets = []
        train_loaders = []
        for file in args.revision_files:
            # train_dataset_temp = SeqDataset("./Data/US/online/{}".format(file), "cases", "./Data/US/online/{}".format("_".join(file.split("_")[:-1])+"_202102.csv"), args.mask_column)
            train_dataset_temp = SeqDataset("./Data/US/online/{}".format(file), "cases", "./Data/US/online/{}".format("_".join(file.split("_")[:-2])+"_202102.csv"), args.mask_column)
            train_datasets.append(train_dataset_temp)
            train_loader_temp = torch.utils.data.DataLoader(train_dataset_temp, batch_size=800, shuffle=False)
            train_loaders.append(train_loader_temp)

        test_dataset = SeqDataset("./Data/US/online/2021-01-16_0_moving_202102.csv".format(args.date), "cases", "./Data/US/online/2021-01-16_0_moving_202113.csv", args.mask_column)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=800, shuffle=False)

        metas_train_dim = 1
        X_train_dim = 15 if args.mask_column == "" else 14
        params['num_steps'] = len(test_dataset)

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

        for (x, y) in train_loaders[0]:
            scaler.fit(x.numpy()[:, 0:1])
        
        ''' step 1-1: training  gradmeta''' 
        if train_flag:
            for tr_idx in range(len(train_loaders)):
                param_model.train()
                param_model.to(devices[0])
                assert param_model != None
                opt = torch.optim.AdamW(param_model.parameters(),lr=CONFIGS[params["disease"]]["learning_rate"])
                loss_fcn = torch.nn.MSELoss(reduction='none')
                best_loss = np.inf
                losses = []
                for epi in tqdm(range(num_epochs)):
                    batch_predictions = []
                    epoch_loss = 0
                    for batch, (x, y) in enumerate(train_loaders[tr_idx]):   
                        abm = build_simulator(params,devices)
                        # forward pass param model
                        meta = torch.Tensor([[1]])
                        # we have 16 age groups
                        counties = torch.ones(16)
                        meta = meta.to(devices[0])
                        x = x.to(devices[0]).unsqueeze(0)
                        y = y.to(devices[0]).unsqueeze(0)
                        # param_values, predictions_2_in_1, lstm_targets = param_model_forward(param_model,params,x,meta)
                        param_values, lstm_targets = param_model_forward(param_model,params,x,meta)

                        # params_epi, seed_status, adjustment_matrix = param_values[0], param_values[1], param_values[2]
                        # forward simulator for several time steps
                        # get predictions after running meta-population model
                        predictions = forward_simulator(params,param_values,abm,training_num_steps,counties,devices)
                        # predictions_2_in_1 = param_model.forward_lstm(predictions, x)

                        # loss weight is set as all-one values                    
                        loss = (loss_fcn(y, predictions[:, :len(train_datasets[tr_idx])])).mean().sqrt()

                        loss.backward()

                        torch.nn.utils.clip_grad_norm_(param_model.parameters(), CLIP)
                        opt.step()
                        opt.zero_grad(set_to_none=True)
                        epoch_loss += torch.sqrt(loss.detach()).item()

                    losses.append(epoch_loss/(batch+1))  # divide by number of batches
                    if (epi + 1) % 10 == 0:
                        tqdm.write("\033[92m" + f"[Epoch {epi + 1}] Loss: {epoch_loss:.4f}" + "\033[0m")

                    if torch.isnan(loss):
                        break
                    ''' save best model '''
                    if epoch_loss < best_loss:
                        save_model(param_model,file_name,params['disease'], args)
                        best_loss = epoch_loss
                        # print("current best loss is {}".format(best_loss))
                    # print('epoch {} time (s): {:.2f}'.format(epi,time.time()- start))


                ''' step 1-2: training  adaptor''' 
                param_model.train()
                param_model.to(devices[0])
                assert param_model != None
                opt = torch.optim.Adam(param_model.lstm_model.parameters(),lr=CONFIGS[params["disease"]]["learning_rate"])
                loss_fcn = torch.nn.MSELoss(reduction='none')
                best_loss = np.inf
                losses = []
                for epi in tqdm(range(int(num_epochs/5))):
                    batch_predictions = []
                    epoch_loss = 0
                    for batch, (x, y) in enumerate(train_loaders[tr_idx]):   
                        abm = build_simulator(params,devices)
                        # forward pass param model
                        meta = torch.Tensor([[1]])
                        # we have 16 age groups
                        counties = torch.ones(16)
                        meta = meta.to(devices[0])
                        x = x.to(devices[0]).unsqueeze(0)
                        y = y.to(devices[0]).unsqueeze(0)
                        # param_values, predictions_2_in_1, lstm_targets = param_model_forward(param_model,params,x,meta)
                        param_values, lstm_targets = param_model_forward(param_model,params,x,meta)
                        # forward simulator for several time steps
                        if BENCHMARK_TRAIN:
                            start_bench = time.time()
                        # get predictions after running meta-population model
                        predictions = forward_simulator(params,param_values,abm,training_num_steps,counties,devices)
                        predictions_2_in_1 = param_model.forward_lstm(predictions, x)

                        # pdb.set_trace()

                        # loss weight is set as all-one values

                        # print(lstm_targets.shape, predictions_2_in_1.shape)
                        loss = loss_fcn(lstm_targets[:, :, 0], predictions_2_in_1[:(len(train_datasets[tr_idx])-4)]).mean()

                        loss.backward()

                        torch.nn.utils.clip_grad_norm_(param_model.parameters(), CLIP)
                        opt.step()
                        opt.zero_grad(set_to_none=True)
                        epoch_loss += torch.sqrt(loss.detach()).item()

                    losses.append(epoch_loss/(batch+1))  # divide by number of batches
                    if (epi + 1) % 10 == 0:
                        tqdm.write("\033[92m" + f"[Epoch {epi + 1}] Loss: {epoch_loss:.4f}" + "\033[0m")

                    if torch.isnan(loss):
                        break
                    ''' save best model '''
                    if epoch_loss < best_loss:
                        save_model(param_model,file_name,params['disease'], args)
                        best_loss = epoch_loss



                ''' step 1-3: train together''' 
                param_model.train()
                param_model.to(devices[0])
                assert param_model != None
                opt = torch.optim.Adam(param_model.parameters(),lr=CONFIGS[params["disease"]]["learning_rate"])
                loss_fcn = torch.nn.MSELoss(reduction='none')
                best_loss = np.inf
                losses = []
                for epi in tqdm(range(int(num_epochs/5))):
                    batch_predictions = []
                    epoch_loss = 0
                    for batch, (x, y) in enumerate(train_loaders[tr_idx]):   
                        abm = build_simulator(params,devices)
                        # forward pass param model
                        meta = torch.Tensor([[1]])
                        # we have 16 age groups
                        counties = torch.ones(16)
                        meta = meta.to(devices[0])
                        x = x.to(devices[0]).unsqueeze(0)
                        y = y.to(devices[0]).unsqueeze(0)
                        # param_values, predictions_2_in_1, lstm_targets = param_model_forward(param_model,params,x,meta)
                        param_values, lstm_targets = param_model_forward(param_model,params,x,meta)
                        # forward simulator for several time steps
                        if BENCHMARK_TRAIN:
                            start_bench = time.time()
                        # get predictions after running meta-population model
                        predictions = forward_simulator(params,param_values,abm,training_num_steps,counties,devices)
                        predictions_2_in_1 = param_model.forward_lstm(predictions, x)

                        # loss weight is set as all-one values
                        loss_weight = torch.ones((len(counties),training_num_steps,1)).to(devices[0])
                        
                        alpha = epi/NUM_EPOCHS_DIFF
                        scale = 700 * (epi+1)/NUM_EPOCHS_DIFF
                    
                        # loss = loss_fcn(lstm_targets[:, :, 0], predictions_2_in_1).mean()
                        # print(y[0].shape, predictions[:, :len(train_datasets[tr_idx])].shape)
                        loss = (1-alpha)*(loss_fcn(y, predictions[:, :len(train_datasets[tr_idx])])).mean().sqrt() / scale + alpha * loss_fcn(lstm_targets[:, :, 0], predictions_2_in_1[:(len(train_datasets[tr_idx])-4)]).mean()

                        loss.backward()

                        torch.nn.utils.clip_grad_norm_(param_model.parameters(), CLIP)
                        opt.step()
                        opt.zero_grad(set_to_none=True)
                        epoch_loss += torch.sqrt(loss.detach()).item()

                    losses.append(epoch_loss/(batch+1))  # divide by number of batches
                    if (epi + 1) % 10 == 0:
                        tqdm.write("\033[92m" + f"[Epoch {epi + 1}] Loss: {epoch_loss:.4f}" + "\033[0m")

                    if torch.isnan(loss):
                        break
                    ''' save best model '''
                    if epoch_loss < best_loss:
                        save_model(param_model,file_name,params['disease'], args)
                        best_loss = epoch_loss
            



        ''' step 2: inference step  ''' 
        ''' upload best model in inference ''' 
        param_model = None; abm = None
        
        param_model = build_param_model(copy(params),metas_train_dim, X_train_dim,devices[0],CUSTOM_INIT=True)
        param_model.eval()
        
        
        # load param model from the saved directory
        print("load model")
        param_model = load_model(param_model,file_name,params['disease'],devices[0], args)
        num_step = training_num_steps # adding number of prediction horizon to the simulation step
        batch_predictions = []
        two_in_one_predictions = []
        counties_predicted = []
        learned_params = []
        with torch.no_grad():
            for batch, (x, y) in enumerate(test_loader):
                abm = build_simulator(params,devices)
                # forward pass param model
                meta = torch.Tensor([[1]])
                counties = torch.ones(16)
                meta = meta.to(devices[0])
                test_x = deepcopy(x)
                test_x = test_x.to(devices[0]).unsqueeze(0)
                
                
                # _, predictions_2_in_1, lstm_targets = param_model_forward(param_model,params,test_x,meta)
                param_values, lstm_targets = param_model_forward(param_model,params,test_x,meta)


                predictions = forward_simulator(params,param_values,abm,num_step,counties,devices)

                predictions_2_in_1 = param_model.forward_lstm(predictions, test_x)

                predictions_2_in_1 = torch.Tensor(scaler.inverse_transform(predictions_2_in_1.detach().cpu().numpy())).to(devices[0])                

                x = x.to(devices[0]).unsqueeze(0)[:, :CONFIGS[params["disease"]]["train_days"], :]
                if 'meta' in params["model_name"]:
                    param_values, _ = param_model_forward(param_model,params,x,meta)
                else:
                    param_values = param_model_forward(param_model,params,x,meta)
                
                # forward simulator for several time steps
                
                preds = forward_simulator(params,param_values,abm,num_step,counties,devices)
                batch_predictions.append(preds)
                two_in_one_predictions.append(predictions_2_in_1.reshape(1, -1, 1))
                counties_predicted.extend(counties)
                
        predictions = torch.cat(batch_predictions,axis=0)
        predictions_two = torch.cat(two_in_one_predictions, axis=0)

        # calculate training and testing RMSE values
        all_rmses = []
        training_mae = []
        all_mapes_train = []
        for state_idx, target_values in enumerate(y.unsqueeze(0)):
            target = torch.squeeze(target_values.detach()).numpy()
            predictions_train = torch.squeeze(predictions.detach()[state_idx]).cpu().numpy()
            rmse = np.sqrt(
            np.mean((target[-NOWCASTING_PERIOD:CONFIGS[params["disease"]]["train_days"]] -
                     predictions_train[-NOWCASTING_PERIOD:CONFIGS[params["disease"]]["train_days"]])**2))
            all_rmses.append(rmse)
            mae_train = np.mean(np.absolute((target[-NOWCASTING_PERIOD:CONFIGS[params["disease"]]["train_days"]] -
                     predictions_train[-NOWCASTING_PERIOD:CONFIGS[params["disease"]]["train_days"]])))

            plot_predictions_nowcasting(x[0, :, 0].cpu().numpy(), target, predictions_train, 0, 0, state_idx, args)
            
            all_mapes_train.append(np.mean(np.abs((target[-NOWCASTING_PERIOD:CONFIGS[params["disease"]]["train_days"]] -
                     predictions_train[-NOWCASTING_PERIOD:CONFIGS[params["disease"]]["train_days"]])) / target[-NOWCASTING_PERIOD:CONFIGS[params["disease"]]["train_days"]]) * 100)

            training_mae.append(mae_train)
        print("GradMeta RMSE: {:.4f}".format(sum(all_rmses)/len(all_rmses)))
        print('GradMeta MAE: {:.4f}'.format(sum(training_mae)/len(training_mae)))
        print('GradMeta MAPES: {:.4f}'.format(sum(all_mapes_train)/len(all_mapes_train)))



        # calculate training and testing RMSE values
        all_rmses = []
        training_mae = []
        all_mapes_train = []
        for state_idx, target_values in enumerate(y.unsqueeze(0)):
            target = torch.squeeze(target_values.detach()).numpy()
            predictions_train = torch.squeeze(predictions_two.detach()[state_idx]).cpu().numpy()
            rmse = np.sqrt(
            np.mean((target[-NOWCASTING_PERIOD:CONFIGS[params["disease"]]["train_days"]] -
                     predictions_train[-NOWCASTING_PERIOD:(CONFIGS[params["disease"]]["train_days"]-4)])**2))
            
            mae_train = np.mean(np.absolute((target[-NOWCASTING_PERIOD:CONFIGS[params["disease"]]["train_days"]] -
                     predictions_train[-NOWCASTING_PERIOD:CONFIGS[params["disease"]]["train_days"]-4])))
            
            args.note += "_w_adapter"
            plot_predictions(target, predictions_train, rmse, 0, state_idx, args, N_lag=4)
            
            all_rmses.append(rmse)
            all_mapes_train.append(np.mean(np.abs((target[-NOWCASTING_PERIOD:CONFIGS[params["disease"]]["train_days"]] -
                     predictions_train[-NOWCASTING_PERIOD:CONFIGS[params["disease"]]["train_days"]-4])) / target[-NOWCASTING_PERIOD:CONFIGS[params["disease"]]["train_days"]]) * 100)
            training_mae.append(mae_train)
        print("GradMeta (w/ errr correction) RMSE: {:.4f}".format(sum(all_rmses)/len(all_rmses)))
        print("GradMeta (w/ errr correction) MAE: {:.4f}".format(sum(training_mae)/len(training_mae)))
        print("GradMeta (w/ errr correction) MAPES: {:.4f}".format(sum(all_mapes_train)/len(all_mapes_train)))



        # we only care about the last predictions
        # predictions are weekly, so we only care about the last 4
        predictions = predictions_two.squeeze(2)[:,-DAYS_HEAD:] 
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

def train_predict(args, configs):
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

    global CONFIGS
    # configs for different datasets
    CONFIGS = configs["configurations"]
    for key, value in CONFIGS.items():
        if "train_days_base" in value:
            value["train_days"] = value["train_days_base"] * args.week

    global BENCHMARK_TRAIN, NUM_EPOCHS_DIFF, DAYS_HEAD, SAVE_MODEL_PATH, SMOOTH_WINDOW
    # if calculating training time
    BENCHMARK_TRAIN = False
    # number of epochs
    NUM_EPOCHS_DIFF = configs["num_epochs_diff"]
    print("---- MAIN IMPORTS SUCCESSFUL -----")



    # define the prediction horizon
    DAYS_HEAD = configs["days_head"]  # 4 weeks ahead

    # define the directory for saving the model
    SAVE_MODEL_PATH = configs["save_model_path"]

    # define the size of smooth window
    SMOOTH_WINDOW= configs["smooth_window"]


    params = {}            
    params['seed'] = args.seed
    params['num_runs'] = args.num_runs
    params['disease'] = args.disease
    params['inference_only'] = args.inference_only
    params['model_name'] = args.model_name
    params['date'] = args.date

        
    verbose = False
    # upper and lower bounds of the neural network prediction
    global MIN_VAL_PARAMS, MAX_VAL_PARAMS, MIN_VAL_PARAMS_2, MAX_VAL_PARAMS_2, NOWCASTING_PERIOD
    MIN_VAL_PARAMS = configs["min_val_params"]
    MAX_VAL_PARAMS = configs["max_val_params"]

    MIN_VAL_PARAMS_2 = configs["min_val_params_2"]
        
    MAX_VAL_PARAMS_2 = configs["max_val_params_2"]
    NOWCASTING_PERIOD = configs['nowcasting_period']
    
    global scaler
    from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
    scaler = MinMaxScaler()

    # start training!
    counties_predicted, predictions, learned_params = runner(params, devices, verbose, args)

    return counties_predicted, predictions, learned_params
    