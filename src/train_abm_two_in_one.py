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
from model_utils import EmbedAttenSeq, fetch_county_data_covid, fetch_county_data_flu, DecodeSeq, SEIRM, SIRS, MetapopulationSEIRM, fetch_age_group_data_covid, moving_average, MetapopulationSEIRMBeta
from visualize_results import *

from models_pets import LSTM_MCDO, LSTM_Two_Encoder
from utils import *
import yaml

def save_model(model,file_name,disease,region,week, args):
    PATH = os.path.join(SAVE_MODEL_PATH,disease,region, args.date)
    if not os.path.exists(PATH):
        os.makedirs(PATH)  
    torch.save(model.state_dict(), PATH+'/' + file_name+' '+week + ".pth")

def load_model(model,file_name,disease,region,week,device, args):
    PATH = os.path.join(SAVE_MODEL_PATH,disease,region, args.date)
    model.load_state_dict(torch.load(PATH+'/' + file_name+' '+week + ".pth",map_location=device))
    return model

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
        # self.out_layer4.apply(init_weights)

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
        self.lstm_model = LSTM_MCDO(1, self.hidden_rnn, self.hidden_dense, self.output_dim, self.activation)

    def forward(self, x, meta, x_2, meta2, train_X, train_Y):
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
        
        outputs = self.lstm_model(train_X)
        # print(outputs.shapes)
        return out, out2, out3, outputs, train_Y

def param_model_forward(param_model,params,x,meta):
    if 'meta' in params['model_name']:
        if 'bogota' in params["disease"]:
            # data and meta2 have been initialized before
            x_ = scaler.transform(x.cpu()[:, :, 0].T)
            reframed = series_to_supervised(x_, 4, 1)
            reframed_data = torch.Tensor(reframed.values)
            reframed_data = torch.unsqueeze(reframed_data, 1)
            train_X, train_Y = reframed_data[:, :, :4].to(devices[0]).permute(0, 2, 1), reframed_data[:, :, -1:].to(devices[0])
            param_prediction, seed_prediction, adjustment_matrix, prediction, targets = param_model.forward(x, meta, data, meta2, train_X, train_Y)  # time-varying
            action_value = [param_prediction, seed_prediction, adjustment_matrix]
        else:
            param_prediction, seed_prediction, adjustment_matrix = param_model.forward(x, meta)  # time-varying
            action_value = [param_prediction, seed_prediction, adjustment_matrix]
    else:
        raise ValueError('model name not valid')
    
    return action_value, prediction, targets

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
    return param_model

def build_simulator(params,devices):
    ''' 
    build simulator: MetaPopulation
    contact matrix and population data are extracted from the public source
    '''

    if 'bogota' in params['disease']:
        population_data = pd.read_csv("Data/Processed/population_bogota.csv")
        flow_data = pd.read_csv("Data/Processed/contact_matrix_bogota.csv", index_col=0)
        abm = MetapopulationSEIRMBeta(params, devices[0], CONFIGS[params['disease']]["num_patch"], torch.Tensor(flow_data.values), torch.Tensor(population_data['Population'].values))
    return abm

def forward_simulator(params,param_values,abm,training_num_steps,counties,devices):
    if params['joint']:
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

        for (x, y) in train_loader:
            scaler.fit(x.numpy()[:, 0:1])
            print("Scaler Prepared")
        
        ''' step 1: training  ''' 
        if train_flag:
            param_model.train()
            param_model.to(devices[0])
            assert param_model != None
            opt = torch.optim.AdamW(param_model.parameters(),lr=CONFIGS[params["disease"]]["learning_rate"],weight_decay=0.01)
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
                    abm = build_simulator(params,devices)
                    # forward pass param model
                    meta = torch.Tensor([[1]])
                    # we have 16 age groups
                    counties = torch.ones(16)
                    meta = meta.to(devices[0])
                    x = x.to(devices[0]).unsqueeze(0)
                    y = y.to(devices[0]).unsqueeze(0)
                    param_values, predictions_2_in_1, lstm_targets = param_model_forward(param_model,params,x,meta)
                    if verbose:
                        if param_values.dim()>2:
                            print(param_values[:,[0,-1],:])
                        else:
                            print(param_values)
                    # forward simulator for several time steps
                    if BENCHMARK_TRAIN:
                        start_bench = time.time()
                    # get predictions after running meta-population model
                    predictions = forward_simulator(params,param_values,abm,training_num_steps,counties,devices)
                    # print(predictions_2_in_1.shape)
                    
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
                    
                    # print(y.shape)
                    
                    alpha = epi/NUM_EPOCHS_DIFF
                    scale = 1000 * (epi+1)/NUM_EPOCHS_DIFF
                 
                    # loss = (1-alpha)*(loss_weight*loss_fcn(y, predictions)).mean() + alpha * loss_fcn(y.squeeze()[4:], predictions_2_in_1).mean()
                    loss = (1-alpha)*(loss_weight*loss_fcn(y, predictions)).mean().sqrt() / scale + alpha * loss_fcn(lstm_targets[:, :, 0], predictions_2_in_1).mean()
                    # print((loss_weight*loss_fcn(y, predictions)).mean().sqrt(), loss_fcn(lstm_targets, predictions_2_in_1).mean())

                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(param_model.parameters(), CLIP)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    epoch_loss += torch.sqrt(loss.detach()).item()


                losses.append(epoch_loss/(batch+1))  # divide by number of batches
                if (epi + 1) % 50 == 0:
                    print("epoch ", epi)
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
                    # print("current best loss is {}".format(best_loss))
                # print('epoch {} time (s): {:.2f}'.format(epi,time.time()- start))

        ''' step 2: inference step  ''' 
        ''' upload best model in inference ''' 
        param_model = None; abm = None
        
        param_model = build_param_model(copy(params),metas_train_dim, X_train_dim,devices[0],CUSTOM_INIT=True)
        param_model.eval()
        
        
        # load param model from the saved directory
        if params['joint']:
            print("load model")
            param_model = load_model(param_model,file_name,params['disease'],'joint',params['pred_week'],devices[0], args)
        else:
            param_model = load_model(param_model,file_name,params['disease'],params['county_id'],params['pred_week'],devices[0], args)

        num_step = training_num_steps + DAYS_HEAD # adding number of prediction horizon to the simulation step
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
                
                
                
                _, predictions_2_in_1, lstm_targets = param_model_forward(param_model,params,test_x,meta)
                predictions_2_in_1 = torch.Tensor(scaler.inverse_transform(predictions_2_in_1.detach().cpu().numpy())).to(devices[0])
                

                x = x.to(devices[0]).unsqueeze(0)[:, :CONFIGS[params["disease"]]["train_days"], :]
                if 'meta' in params["model_name"]:
                    param_values, _, _ = param_model_forward(param_model,params,x,meta)
                else:
                    param_values = param_model_forward(param_model,params,x,meta)
                
                # forward simulator for several time steps
                
                preds = forward_simulator(params,param_values,abm,num_step,counties,devices)
                batch_predictions.append(preds)
                two_in_one_predictions.append(predictions_2_in_1.reshape(1, -1, 1))
                counties_predicted.extend(counties)
                if 'meta' in params["model_name"]:
                    learned_params.extend(np.array(param_values[0].cpu().detach()))
                    learned_params.extend(np.array([param_values[1].cpu().detach()]))
                    learned_params.extend(np.array([param_values[2].cpu().detach()]))
                else:
                    learned_params.extend(np.array(param_values.cpu().detach()))
                
        predictions = torch.cat(batch_predictions,axis=0)
        predictions_two = torch.cat(two_in_one_predictions, axis=0)

        # calculate training and testing RMSE values
        all_rmses = []
        all_maes = []
        all_mapes = []
        training_mae = []
        all_mapes_train = []
        for state_idx, target_values in enumerate(y.unsqueeze(0)):
            target = torch.squeeze(target_values.detach()).numpy()
            predictions_train = torch.squeeze(predictions.detach()[state_idx]).cpu().numpy()
            rmse = np.sqrt(
            np.mean((target[:CONFIGS[params["disease"]]["train_days"]] -
                     predictions_train[:CONFIGS[params["disease"]]["train_days"]])**2))
            rmse_test = np.sqrt(
            np.mean((target[CONFIGS[params["disease"]]["train_days"]:(CONFIGS[params["disease"]]["train_days"]+CONFIGS[params["disease"]]["test_days"])] -
                     predictions_train[CONFIGS[params["disease"]]["train_days"]:(CONFIGS[params["disease"]]["train_days"]+CONFIGS[params["disease"]]["test_days"])])**2))
            mae_train = np.mean(np.absolute((target[4:CONFIGS[params["disease"]]["train_days"]] -
                     predictions_train[4:CONFIGS[params["disease"]]["train_days"]])))
            
            mae_test = np.mean(np.absolute(target[CONFIGS[params["disease"]]["train_days"]:(CONFIGS[params["disease"]]["train_days"]+CONFIGS[params["disease"]]["test_days"])] -
                     predictions_train[CONFIGS[params["disease"]]["train_days"]:(CONFIGS[params["disease"]]["train_days"]+CONFIGS[params["disease"]]["test_days"])]))

            plot_predictions(target, predictions_train, rmse, rmse_test, state_idx, args)
            
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



        # calculate training and testing RMSE values
        all_rmses = []
        all_maes = []
        all_mapes = []
        training_mae = []
        all_mapes_train = []
        for state_idx, target_values in enumerate(y.unsqueeze(0)):
            target = torch.squeeze(target_values.detach()).numpy()
            predictions_train = torch.squeeze(predictions_two.detach()[state_idx]).cpu().numpy()
            print(target.shape, predictions_train.shape)
            # print(target[(CONFIGS[params["disease"]]["train_days"]-10):CONFIGS[params["disease"]]["train_days"]], predictions_train[(CONFIGS[params["disease"]]["train_days"]-10):CONFIGS[params["disease"]]["train_days"]])
            print(target[:CONFIGS[params["disease"]]["train_days"]].shape, predictions_train[:CONFIGS[params["disease"]]["train_days"]].shape)
            rmse = np.sqrt(
            np.mean((target[4:CONFIGS[params["disease"]]["train_days"]] -
                     predictions_train[:(CONFIGS[params["disease"]]["train_days"]-4)])**2))
            rmse_test = np.sqrt(
            np.mean((target[CONFIGS[params["disease"]]["train_days"]:(CONFIGS[params["disease"]]["train_days"]+CONFIGS[params["disease"]]["test_days"])] -
                     predictions_train[(CONFIGS[params["disease"]]["train_days"]-4):(CONFIGS[params["disease"]]["train_days"]+CONFIGS[params["disease"]]["test_days"]-4)])**2))
            mae_train = np.mean(np.absolute((target[4:CONFIGS[params["disease"]]["train_days"]] -
                     predictions_train[4:CONFIGS[params["disease"]]["train_days"]])))
            
            mae_test = np.mean(np.absolute(target[CONFIGS[params["disease"]]["train_days"]:(CONFIGS[params["disease"]]["train_days"]+CONFIGS[params["disease"]]["test_days"])] -
                     predictions_train[(CONFIGS[params["disease"]]["train_days"]-4):(CONFIGS[params["disease"]]["train_days"]+CONFIGS[params["disease"]]["test_days"]-4)]))

            args.note = "LSTM"
            plot_predictions(target, predictions_train, rmse, rmse_test, state_idx, args)
            
            all_rmses.append(rmse_test)
            all_maes.append(mae_test)
            all_mapes.append(np.mean(np.abs((target[CONFIGS[params["disease"]]["train_days"]:(CONFIGS[params["disease"]]["train_days"]+CONFIGS[params["disease"]]["test_days"])] -
                     predictions_train[(CONFIGS[params["disease"]]["train_days"]-4):(CONFIGS[params["disease"]]["train_days"]+CONFIGS[params["disease"]]["test_days"]-4)])) / target[CONFIGS[params["disease"]]["train_days"]:(CONFIGS[params["disease"]]["train_days"]+CONFIGS[params["disease"]]["test_days"])]) * 100)
            all_mapes_train.append(np.mean(np.abs((target[4:CONFIGS[params["disease"]]["train_days"]] -
                     predictions_train[4:CONFIGS[params["disease"]]["train_days"]])) / target[4:CONFIGS[params["disease"]]["train_days"]]) * 100)
            training_mae.append(mae_train)
        print('RMSE: ', sum(all_rmses)/len(all_rmses))
        print(all_maes)
        print('testing MAE: ', sum(all_maes)/len(all_maes))
        print('training MAE: ', sum(training_mae)/len(training_mae))
        print('training MAPES', sum(all_mapes_train)/len(all_mapes_train))
        print('testing MAPES: ', sum(all_mapes)/len(all_mapes))



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
    

    global data, meta2
    # load the pre-processed transaction dataset and normalize the dataset
    data = torch.load(configs["data_path"].format(args.date)).to(torch.float32).unsqueeze(2)
    data = torch.nn.functional.normalize(data,dim=0).to(devices[0])
    meta2 = torch.eye(data.shape[0]).to(devices[0])

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
    MIN_VAL_PARAMS = configs["min_val_params"]
    MAX_VAL_PARAMS = configs["max_val_params"]

    MIN_VAL_PARAMS_2 = configs["min_val_params_2"]
        
    MAX_VAL_PARAMS_2 = configs["max_val_params_2"]
    
    global scaler
    from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
    scaler = MinMaxScaler()

    # start training!
    counties_predicted, predictions, learned_params = runner(params, devices, verbose, args)

    return counties_predicted, predictions, learned_params
    