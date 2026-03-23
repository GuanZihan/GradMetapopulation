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
# parser.add_argument('--test', action='store_true')
parser.add_argument('--root', type=str, default="./Data/Processed")
parser.add_argument('--saved_root', type=str, default="./Data/Processed/online")
parser.add_argument('--week', type=int, default=49)
parser.add_argument('--case_only', action="store_true")
parser.add_argument('--revision', type=str, default="2020-08-30")

args = parser.parse_args()

start_date = pd.to_datetime("2020-04-12")



period = pd.to_timedelta((int(args.week)+args.moving_window)*7, unit='D') # 350

end_date = start_date + period


# data = pd.read_csv(os.path.join(args.root, "result_2021-02-28.csv"))
data = pd.read_csv(os.path.join(args.root, "result_2021-03-27.csv"))


# print(data["date"])
if "Unnamed: 0" in data.columns:
    data.drop(columns="Unnamed: 0", inplace=True)
data["date"] = pd.to_datetime(data.date, format='%Y-%m-%d')
data_selected = data[(data["date"] <= end_date) & (data["date"] >= start_date)]
data_selected = data_selected[["date", args.revision + "_version"]][(data["date"] <= pd.to_datetime(args.revision))]
# print(data_selected)
# input()
data_selected = data_selected.rename(columns={args.revision + "_version": 'cases'})

GHT = pd.read_csv(os.path.join(args.root,"multiTimeline.csv"))
GHT['Week'] = pd.to_datetime(GHT.Week, format='%Y-%m-%d')
GHT = GHT[(GHT["Week"] <= end_date) & (GHT["Week"] >= start_date)]

GHT = GHT.set_index('Week').resample('D').ffill().reset_index()

pcr_data = pd.read_csv(os.path.join(args.root,"Pruebas_PCR_procesadas_de_COVID-19_en_Colombia__Departamental__20240910.csv"))

pcr_data['Fecha'] = pd.to_datetime(pcr_data.Fecha, format='%Y-%m-%d')
# data = data.set_index('Fecha')
pcr_data = pcr_data[["Fecha", "Bogota"]]
pcr_data['incremental'] = pcr_data['Bogota'].diff()
pcr_data.drop(columns="Bogota", inplace=True)

pcr_data = pcr_data[(pcr_data["Fecha"] <= end_date) & (pcr_data["Fecha"] >= start_date)]
# pcr_data.drop(columns="Fecha", inplace=True)
pcr_data = pcr_data.fillna(0)
print(data_selected.shape)
print(GHT.shape)
print(pcr_data.shape)

df = pd.read_csv("./Data/Processed/Global_Mobility_Report.csv", low_memory=False)

df_co_1 = region_filter(["CO"], ["Bogota"], "")

mobility_data = df_co_1[['date', 'retail_and_recreation_percent_change_from_baseline',
       'grocery_and_pharmacy_percent_change_from_baseline',
       'parks_percent_change_from_baseline',
       'transit_stations_percent_change_from_baseline',
       'workplaces_percent_change_from_baseline',
       'residential_percent_change_from_baseline']]
mobility_data["date"] = pd.to_datetime(mobility_data["date"], format='%Y-%m-%d')

merged_dataset = pd.merge(pd.merge(pd.merge(data_selected,GHT,left_on='date', right_on="Week"),pcr_data, left_on='date', right_on="Fecha"), mobility_data, left_on="date", right_on="date")
print(merged_dataset["date"])

# print("Shape of Two Files: ", merged_dataset["cases"].shape, pd.read_csv("Data/Processed/aggregated_{}.csv".format("test" if args.test else "train"))["Cases"].shape)
# merged_dataset["cases"] = pd.read_csv("Data/Processed/aggregated_{}.csv".format("test" if args.test else "train"))["Cases"]

if not args.case_only:
    merged_dataset.drop(columns=["Week", "Fecha", "incremental", "covid-19 vacuna"], inplace=True)
else:
    merged_dataset.drop(columns=["Fecha", "incremental", "covid-19 vacuna"], inplace=True)
    merged_dataset = merged_dataset[["Week", "cases"]]
    # print(merged_dataset)
    # input()
    # merged_dataset.loc["index"] = 0
    merged_dataset.set_index('Week', inplace=True)
    merged_dataset = merged_dataset.T
    # merged_dataset = merged_dataset.reset_index()
    



if not os.path.exists(args.saved_root):
    os.mkdir(args.saved_root)

merged_dataset = merged_dataset.iloc[:(args.week*7), :]
if not args.case_only:
    dataset_end_date = merged_dataset["date"].iloc[-1].strftime('%Y-%m-%d')
    merged_dataset.drop(columns=["date"], inplace=True)
else:
    dataset_end_date = merged_dataset.columns[-1].date()

file_name = "{}_{}_moving{}_{}.csv".format(dataset_end_date, args.moving_window, "_lstm" if args.case_only else "", ""+args.revision)
print(file_name)
merged_dataset.to_csv(os.path.join(args.saved_root, file_name), index=None)

print(merged_dataset.shape[0])

if not args.case_only:
    assert args.week == merged_dataset.shape[0] / 7
