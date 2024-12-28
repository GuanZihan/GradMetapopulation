import pandas as pd
import numpy as np
import torch
import antropy as ant
from math import exp, sqrt
from scipy.special import erf
import os


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def predict_n_point(model, X, steps, scaler, len_scal, nth_scal, n_back, n_feature=1):
    ### N step ahead forecasting (point estimate) ###
    points = []
    p = torch.Tensor(X).permute(0,2,1).cuda()
    outputs = [i.item() for i in X[0]]
    for i in range(steps):
        pred = model(torch.Tensor(outputs[-3:]).reshape((1,1,3)).cuda())
        pred = pred.detach().cpu().numpy()
        tran_pd = np.asarray([pred[0]]*len_scal).reshape(1,len_scal)
        point = scaler.inverse_transform(tran_pd)
        
        points.append(point[0][nth_scal])
        outputs.append(pred.item())
        
    return points

# define the moving average function
def moving_average(x, w):
    return pd.Series(x).rolling(w, min_periods=1).mean().values

def save_params(
    disease: str,
    model_name: str,
    pred_week: str,
    param_values: np.array,
    args
    ):
    """
        Given an array w/ predictions, save as csv
    """
    
    path = './Results/{}/{}'.format(disease, args.date)
    if not os.path.exists(path):
        os.makedirs(path)
    file_name = '/params_{}_{}.csv'.format(model_name,pred_week)
    # for three predictions made by NN, saving them respectively
    with open(path+file_name, "ab") as f:
        np.savetxt(f, param_values[:-2], delimiter=',')
        np.savetxt(f, param_values[-2], delimiter=',')
        np.savetxt(f, param_values[-1], delimiter=',')