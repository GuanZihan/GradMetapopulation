import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os


def region_filter(country_code, region_1, region_2, all=False):
    mask = ((df["country_region_code"].isin(country_code)) & (df["sub_region_1"].isin(region_1)))
    if all:
        mask = (mask & (df["sub_region_2"].isin(region_2)))
    return df[mask]
    
parser = argparse.ArgumentParser()
parser.add_argument('--moving_window', type=int, default=0)
parser.add_argument('--test', action='store_true')
parser.add_argument('--root', type=str, default="./Data/Processed")
parser.add_argument('--saved_root', type=str, default="./Data/Processed/online")
parser.add_argument('--week', type=int, default=49)
parser.add_argument('--case_only', action="store_true")

args = parser.parse_args()

start_date = pd.to_datetime("2020-04-12") # "2020-03-03"
test_period = 28 if args.test else 0
period = pd.to_timedelta((int(args.week+1)+args.moving_window)*7 + test_period, unit='D') # 350

end_date = start_date + period


data = pd.read_csv(os.path.join(args.root, "minimal_dataset.csv"))
data = data[data["region"] == "Bogota"]
data.drop(columns="Unnamed: 0", inplace=True)
data["date"] = pd.to_datetime(data.date, format='%Y-%m-%d')
data_selected = data[(data["date"] <= end_date) & (data["date"] >= start_date)]

GHT = pd.read_csv(os.path.join(args.root,"multiTimeline.csv"))
GHT['Week'] = pd.to_datetime(GHT.Week, format='%Y-%m-%d')
GHT = GHT[(GHT["Week"] <= end_date) & (GHT["Week"] >= start_date)]

GHT = GHT.set_index('Week').resample('D').ffill().reset_index()

df = pd.read_csv("./Data/Processed/Global_Mobility_Report.csv", low_memory=False)

df_co_1 = region_filter(["CO"], ["Bogota"], "")

mobility_data = df_co_1[['date', 'retail_and_recreation_percent_change_from_baseline',
       'grocery_and_pharmacy_percent_change_from_baseline',
       'parks_percent_change_from_baseline',
       'transit_stations_percent_change_from_baseline',
       'workplaces_percent_change_from_baseline',
       'residential_percent_change_from_baseline']]
mobility_data["date"] = pd.to_datetime(mobility_data["date"], format='%Y-%m-%d')

merged_dataset = pd.merge(
    pd.merge(data_selected, GHT, left_on="date", right_on="Week"),
    mobility_data,
    left_on="date",
    right_on="date",
)

if not args.case_only:
    merged_dataset.drop(columns=["Week", "date", "year", "epi_week", "region", "covid-19 vacuna"], inplace=True)
else:
    merged_dataset.drop(columns=["date", "year", "epi_week", "region", "covid-19 vacuna"], inplace=True)
    merged_dataset = merged_dataset[["Week", "cases"]]
    merged_dataset.set_index('Week', inplace=True)
    merged_dataset = merged_dataset.T
    


if not os.path.exists(args.saved_root):
    os.mkdir(args.saved_root)

merged_dataset = merged_dataset.iloc[:(args.week*7+test_period), :]
merged_dataset.to_csv(os.path.join(args.saved_root, "{}_{}_moving{}.csv".format("test" if args.test else "train", args.moving_window, "_lstm" if args.case_only else "")), index=None)

if not args.case_only:
    assert (args.week + test_period//7) == merged_dataset.shape[0] / 7
