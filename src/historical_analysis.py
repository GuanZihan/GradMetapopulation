import pandas as pd
import os
from tqdm import tqdm

root = "./downloads"
files = os.listdir(root)
histroy = []
start_date = pd.to_datetime("2020-03-03")
for file in tqdm(files):
    try:
        data = pd.read_excel(os.path.join(root, file))
        end_date = pd.to_datetime(file.split(".")[0])
        full_date_range = pd.date_range(start=start_date, end=end_date)
        result_df = pd.DataFrame({'timestamp': full_date_range})
        result_df['count'] = 0  # Initialize count column to 0
        # Count the number of rows for each date in the data
        for date, items in data.groupby(["Fecha Not"]):
            result_df.loc[result_df['timestamp'] == date[0], 'count'] = items.shape[0]
        histroy.append({file: result_df})
    except ValueError:
        print("Error File: ", file)

print(len(histroy))
