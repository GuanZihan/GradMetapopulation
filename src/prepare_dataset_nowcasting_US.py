import argparse
import os

import pandas as pd


def region_filter(df, country_code):
    mask = df["country_region_code"].isin(country_code) & df["sub_region_1"].isna()
    return df[mask]
    
parser = argparse.ArgumentParser()
parser.add_argument('--moving_window', type=int, default=0)
# parser.add_argument('--test', action='store_true')
parser.add_argument('--root', type=str, default="./Data/US")
parser.add_argument('--saved_root', type=str, default="./Data/US/online")
parser.add_argument('--week', type=int, default=49)
parser.add_argument('--case_only', action="store_true")
parser.add_argument('--revision', type=str, default="2020-08-30")

args = parser.parse_args()

start_date = pd.to_datetime("2020-04-12")



period = pd.to_timedelta((int(args.week)+args.moving_window)*7, unit='D') # 350

end_date = start_date + period


# data = pd.read_csv(os.path.join(args.root, "result_2021-02-28.csv"))
data = pd.read_csv(os.path.join(args.root, "result_US.csv"))


# print(data["date"])
if "Unnamed: 0" in data.columns:
    data.drop(columns="Unnamed: 0", inplace=True)
data["date"] = pd.to_datetime(data.date, format='%Y-%m-%d')
data_selected = data[(data["date"] <= end_date) & (data["date"] >= start_date)]
data_selected = data_selected[["date", "vEW{}".format(args.revision)]]
data_selected = data_selected.rename(columns={"vEW{}".format(args.revision): 'cases'})

GHT = pd.read_csv(os.path.join(args.root,"multiTimeline.csv"))
GHT['Week'] = pd.to_datetime(GHT.Week)
GHT = GHT[(GHT["Week"] <= end_date) & (GHT["Week"] >= start_date)]

feature_columns = ["date", "onVentilatorCurrently", "inIcuCurrently", "deathIncrease", "recovered", "hospitalizedIncrease"]
other_features = pd.read_csv(os.path.join(args.root, "covid_data", "covid-hospitalization-all-state-merged_vEW{}.csv".format(args.revision)), parse_dates=["date"], infer_datetime_format=True)
other_features = other_features[other_features.fips == "US"]
other_features = other_features[feature_columns]
# other_features.date = pd.to_datetime(other_features, format='%Y-%m-%d')
other_features = other_features[(other_features["date"] <= end_date) & (other_features["date"] >= start_date)]
print(other_features.shape)
print(data_selected.shape)
print(GHT.shape)

df = pd.read_csv("./Data/Processed/Global_Mobility_Report.csv", low_memory=False)

df_co_1 = region_filter(df, ["US"])

mobility_data = df_co_1[['date', 'retail_and_recreation_percent_change_from_baseline',
       'grocery_and_pharmacy_percent_change_from_baseline',
       'parks_percent_change_from_baseline',
       'transit_stations_percent_change_from_baseline',
       'workplaces_percent_change_from_baseline',
       'residential_percent_change_from_baseline']]
mobility_data["date"] = pd.to_datetime(mobility_data["date"], format='%Y-%m-%d')

merged_dataset = pd.merge(pd.merge(pd.merge(data_selected,GHT,left_on='date', right_on="Week"), other_features, left_on="date", right_on="date"), mobility_data, left_on="date", right_on="date")

if not args.case_only:
    merged_dataset.drop(columns=["Week"], inplace=True)
else:
    # merged_dataset.drop(columns=["Week"], inplace=True)
    # print(merged_dataset)
    # input()
    merged_dataset = merged_dataset[["date", "cases"]]
    # merged_dataset.loc["index"] = 0
    merged_dataset.set_index('date', inplace=True)
    merged_dataset = merged_dataset.T
    # merged_dataset = merged_dataset.reset_index()
    # print(merged_dataset)

if not os.path.exists(args.saved_root):
    os.mkdir(args.saved_root)

merged_dataset = merged_dataset.iloc[:(args.week*7), :]

if not args.case_only:
    dataset_end_date = merged_dataset["date"].iloc[-1].strftime('%Y-%m-%d')
    merged_dataset.drop(columns=["date"], inplace=True)
else:
    dataset_end_date = merged_dataset.columns[-1].date()
other_features.to_csv(os.path.join(args.saved_root, "other_features{}.csv".format("_lstm" if args.case_only else "")), index=None)

file_name = "{}_{}_moving{}_{}.csv".format(dataset_end_date, args.moving_window, "_lstm" if args.case_only else "", ""+args.revision)
merged_dataset.to_csv(os.path.join(args.saved_root, file_name), index=None)