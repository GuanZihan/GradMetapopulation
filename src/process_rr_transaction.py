import pandas as pd
import numpy as np
from tqdm import tqdm
import torch

start_date = pd.to_datetime("2020-03-03")
period = pd.to_timedelta(365, unit='D')
end_date = start_date + period


data = pd.read_csv("./Data/Processed/inperson_df_eps007.csv", index_col=0)

data["date"] = pd.to_datetime(data["date"], format='%Y-%m-%d')
data["Perturbed_zip"] = data["Perturbed_zip"].apply(str)
data = data[data["Perturbed_zip"].str.startswith("11")]
print(data.shape)
# data.to_csv("bogota_transaction.csv")
data_selected = data[(data["date"] <= end_date) & (data["date"] >= start_date)]



all_location_time_series = pd.DataFrame([])
ret = []
for g_idx, item in tqdm(data_selected.groupby(["Perturbed_zip"])):
    all_time_series = []
    for g_idx_2, item2 in item.groupby(["date"]):
        zip_code_time_series = {}
        average_amount_sum = item2["scaled_spendamt"].sum()
        zip_code_time_series["date"] = g_idx_2
        zip_code_time_series["spendamt"] = average_amount_sum
        zip_code_time_series["zipcode"] = g_idx
        all_time_series.append(zip_code_time_series)
    all_time_series = pd.DataFrame(all_time_series)
    all_time_series = all_time_series.set_index('date').resample('D').ffill().reset_index()
    
    all_location_time_series = pd.concat((all_time_series, all_location_time_series))
    print(all_time_series["spendamt"].values.shape)
    input()
    ret.append(all_time_series["spendamt"].values)

# print(g_idx)    
# print(all_location_time_series.shape)
# all_location_time_series.to_csv("./Data/Processed/transaction_dataset_private_final.csv", index=None)

torch.save(torch.from_numpy(np.array(ret)), "./Data/Processed/transaction_private_final.pt")