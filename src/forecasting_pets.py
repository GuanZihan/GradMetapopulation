import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from datetime import datetime,timedelta
from sklearn.cluster import KMeans
from datetime import datetime
from epiweeks import Week, Year

from utils import *
from models_pets import LSTM_MCDO, LSTM_Two_Encoder, LSTM_Three_Encoder

from torchsummary import summary
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import random_split
import torch.nn as nn


import argparse
import re
import pdb
import json
import yaml
import os

import matplotlib.pyplot as plt


def get_week(date, weeks):
    for week in weeks:
        s,e = week.split('_')
        if s <= date and date <= e:
            return week

def predict_multi_hop(model, criterion, train_data, val_data, epoch, args, training_loss):
    model.eval()

    current_input = torch.Tensor(train_data[0])[train_size-1:train_size]
    num_steps_ahead = n_ahead
    print("Nums of Predictions Ahead: ", num_steps_ahead)
    current_input = current_input.cuda() # 1, 7, 1
    all_predictions = [scaler.inverse_transform(current_input[:, :, 0].cpu().numpy())[0, -1].item()]
    print(torch.Tensor(train_data[1])[train_size-1:train_size].detach().numpy())
    all_targets = scaler.inverse_transform(torch.Tensor(train_data[1])[train_size-1:train_size].detach().numpy()[0])
    predictions = model(current_input.permute(0, 2,1).cuda(), t_data, meta)
    predictions = scaler.inverse_transform(predictions.detach().cpu().numpy()).tolist()[0]
    all_predictions += predictions

    plt.figure()
    plt.plot(scaler.inverse_transform(train_data[1][:, 0]))
    plt.plot(range(train_size-1, train_size+num_steps_ahead), all_predictions)

    all_predictions = np.array(all_predictions[1:])
    all_targets = np.array(all_targets)
    assert len(all_predictions) == len(all_targets)
    print("MAE: ", np.mean(np.abs(all_predictions - all_targets)))
    print("MAPE: ", np.mean(np.abs(all_predictions - all_targets)/all_targets) * 100)
    print("RMSE: ", np.sqrt(np.mean((all_predictions - all_targets)**2)))

    plt.savefig("Figure-Prediction/lstm/predictions_{}.png".format(args.date))
    


def predict(model, criterion, train_data, val_data, epoch, args, training_loss):
    model.eval()

    current_input = torch.Tensor(train_data[0])[train_size-1:train_size]
    num_steps_ahead = len(val_data)
    print("Nums of Predictions Ahead: ", num_steps_ahead)
    current_input = current_input.cuda() # 1, 7, 1
    all_predictions = [scaler.inverse_transform(current_input[:, :, 0].cpu().numpy())[0, -1].item()]
    all_targets = []
    for step in range(num_steps_ahead):
        predictions = model(current_input.permute(0, 2,1).cuda(), t_data, meta)
        current_input = torch.concat([current_input[:, 1:, :], torch.unsqueeze(predictions, dim=0)], dim=1)
        predictions = scaler.inverse_transform(predictions.detach().cpu().numpy()).item()
        targets = scaler.inverse_transform(val_data[step, :, :].detach().cpu().numpy()).item()
        all_predictions.append(predictions)
        all_targets.append(targets)
    
    plt.figure()
    plt.plot(scaler.inverse_transform(train_data[1][:, 0]))
    plt.plot(range(len(train_data[1])-num_steps_ahead-1, len(train_data[1])), all_predictions)

    all_predictions = np.array(all_predictions[1:])
    all_targets = np.array(all_targets)
    assert len(all_predictions) == len(all_targets)
    print("MAE: ", np.mean(np.abs(all_predictions - all_targets)))
    print("MAPE: ", np.mean(np.abs(all_predictions - all_targets)/all_targets) * 100)
    print("RMSE: ", np.sqrt(np.mean((all_predictions - all_targets)**2)))

    plt.savefig("Figure-Prediction/lstm/predictions_{}.png".format(args.date))
    return
def evaluate(model, criterion, val_data, epoch, args, training_loss):
    model.eval()
    
    record = ()
    
    total_loss = 0
    for inputs, targets in val_data:
        targets = targets[:,:,0].cuda()
        outputs = model(inputs.permute(0, 2,1).cuda(), t_data, meta)
        # outputs = model(inputs.permute(0, 2,1).cuda())
        loss = criterion(outputs, targets)
        total_loss += loss.item()
    
        if (epoch+1) % args.num_epochs == 0:
            print("Predictions\tTargets\tDifference")
            
            horizon = 4
            len_scal = 1
            nth_scal = 0
            trans_ouputs = scaler.inverse_transform(outputs.detach().cpu().numpy())
            trans_targets = scaler.inverse_transform(targets.cpu().numpy())

            print("MAE: ", np.mean(np.abs(trans_ouputs - trans_targets)))
            print("MAPE: ", np.mean(np.abs(trans_ouputs - trans_targets)/trans_targets) * 100)
            print("RMSE: ", np.sqrt(np.mean((trans_ouputs - trans_targets)**2)))
            
            plt.figure()
            plt.plot(trans_targets)
            plt.plot(trans_ouputs)
            
            plt.xlabel("TimeStamp")
            plt.ylabel("# Infections")

            plt.legend(["ground-truth", "predictions"])
            plt.savefig("Figure-Prediction/lstm/outputs_targets_{}.png".format(args.date))
        
            # for idx in range(len(outputs)):
            #     print(str(round(trans_ouputs[idx].item(), 4)) + "\t" + str(round(trans_targets[idx].item(), 4)) + "\t" + str(round(trans_ouputs[idx].item() - trans_targets[idx].item(), 4)))
    
    avg_loss = total_loss / len(val_data)
    if (epoch+1) % args.num_epochs == 0:
        print(f'Validation Loss: {avg_loss:.4f}')
        with open("log.txt", "a+") as f:
            f.write(str(args) + " validation loss " + str(avg_loss) + "\n")
            f.write(str(trans_ouputs[:15]) + "\n" + str(trans_targets[:15]) + "\n")
        record = (avg_loss, training_loss)
    else:
        record = (avg_loss, training_loss)
     
    return record

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="model name")
    parser.add_argument("--data", type=str, default="online/train_0_moving_lstm.csv")
    parser.add_argument("--config_file", type=str, default="src/pet.yml")
    parser.add_argument("--epsilon", type=float, default=8)
    parser.add_argument("--num_epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--index_column", type=str, default='index')
    parser.add_argument("--date", type=str, default="")
    args = parser.parse_args()

    if not os.path.exists("Figure-Prediction/lstm"):
        os.mkdir("Figure-Prediction/lstm")
    
    torch.manual_seed(args.seed)
    
    # extract arguments
    model_name = args.model

    
    
    # Confirmed cases
    data=pd.read_csv(os.path.join("Data/Processed", args.data))
    dates=data.columns
    train_end_date = dates[-1]
    # data=data.set_index(args.index_column)
    for dt in data.columns:
        data=data.rename(columns={dt:dt+'_'+(datetime.strptime(dt,'%Y-%m-%d')+timedelta(days=6)).strftime('%Y-%m-%d')})
    dates = data.columns.values.tolist()
    data = data.T.astype('float')

    

    def moving_average(x, w):
        return pd.Series(x).rolling(w, min_periods=1).mean()
    
    

    n_feature = len(data.columns.tolist())
    # print(data[0].ravel())

    data_ = pd.DataFrame({})

    
    
    for i in range(n_feature):
        moving_averaged = pd.DataFrame(moving_average(data.iloc[:-28, i].ravel(), 7))
        combined_column = pd.concat([moving_averaged, data.iloc[-28:, i].reset_index(drop=True)], ignore_index=True)
        # Print the final shape of the concatenated column
        data_[i] = combined_column
    data_.index = data.index
    data = data_
    
    global t_data, meta
    t_data = torch.load("./Data/Processed/online/transaction_private_lap_{}.pt".format(args.date)).to(torch.float32).unsqueeze(2)
    t_data = torch.nn.functional.normalize(t_data,dim=0).to("cuda:0")
    meta = torch.eye(t_data.shape[0]).to("cuda:0")

    print("Number of Features: \n" + str(n_feature))
    
    train = data[data.index<=train_end_date] # train dataset splitting given a state and a train date
    values = train.values
    scaler = MinMaxScaler()
    scaler.fit(values[:, 0].reshape(-1,1))
    values[:, 0] = scaler.transform(values[:, 0].reshape(-1,1))[:,0]

    n_back = 14
    n_ahead = 7
    n_in = n_back * n_feature
    n_out = n_ahead * n_feature
    
    reframed = series_to_supervised(values, n_back, n_ahead) # chage ts data to supervised form
    reframed_data = reframed.values        
    print("Data Shape is ", reframed_data.shape)
    
    train_X, train_Y = reframed_data[:, :n_in], reframed_data[:, -n_out:]
    train_X = train_X.reshape((train_X.shape[0], n_back, n_feature)) # 333 * 7 * 1
    train_Y = train_Y.reshape(-1,n_ahead,1) # 111 * 3 * 1
    

    print('Building model...')
    input_shape = n_back
    hidden_rnn = 32
    hidden_dense = 16
    output_dim = n_ahead
    activation = 'relu'
    model = LSTM_Two_Encoder(input_shape, hidden_rnn, hidden_dense, output_dim, activation)

    
    train_size = train_X.shape[0] - 28
    train_data = TensorDataset(torch.Tensor(train_X)[:train_size], torch.Tensor(train_Y)[:train_size])
    val_data = TensorDataset(torch.Tensor(train_X)[:], torch.Tensor(train_Y)[:])

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    
    model = model.cuda()
    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    records = []
    model = model.train()      
    
    model = model.cuda()
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0

        for inputs, targets in train_loader:
            targets = targets[:,:,0].cuda()
            inputs = inputs.permute(0,2,1).cuda()
            optimizer.zero_grad()
            outputs = model(inputs, t_data, meta)
            # outputs = model(inputs)
            loss = criterion(outputs, targets) 
            loss.backward()
                
            optimizer.step()
            total_loss += loss.item()


        avg_loss = total_loss / len(train_loader)
        record = evaluate(model, criterion, val_loader, epoch, args, avg_loss)
        
        records.append(record)
        
        if (epoch+1) % 20 == 0:
            print(f'Epoch [{epoch + 1}/{args.num_epochs}], Training Loss: {avg_loss:.4f}')
    
    
    # print(len(records[0]), len(records[1]))
    plt.figure()
    
    plt.plot(np.array(records)[:, 1])
    plt.plot(np.array(records)[:, 0])
    
    
    
    
    plt.savefig("Figure-Prediction/lstm/{}_loss.png".format(args.date))
    
    model.eval()

    # print(data_.shape, train_X.shape, train_Y.shape) # (343, 1) -> (326, 7, 1) (326,10,1)
    print(scaler.transform(data_[-15:]))
    # print(train_X[-2:])
    # print(train_Y[-1])
    # input()

    predict_multi_hop(model, criterion, (train_X, train_Y), torch.Tensor(train_Y)[train_size:], epoch, args, avg_loss)
    print("++++++++++++finish++++++++++++")