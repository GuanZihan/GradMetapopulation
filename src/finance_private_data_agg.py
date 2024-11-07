# Dung Nguyen
# Implement color sampling algorithm
import numpy as np
import pandas as pd
import sys
import math
import random
import csv


def agg_data(data, epsilon, C):

    data["spendamt_clipped"] = data["spendamt"].map(lambda x: min(x, C))
    print(data.head())
    data["merch_postal_code"] = data["merch_postal_code"].apply(str)
    data = data[(data.merch_postal_code.str.startswith("11"))]
    print(data.head())
    private_data = data.groupby(["date", "merch_postal_code"])["spendamt_clipped"].sum().transform(lambda x: x + np.random.laplace(0, 627 * C / epsilon))
    return private_data


def main():
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    epsilon = float(sys.argv[3])
    C = float(sys.argv[4])

    csvfile = open(output_file, 'a')
    result_writer = csv.writer(csvfile, delimiter=',',
                       quotechar='|', quoting=csv.QUOTE_MINIMAL)

    original_data = pd.read_csv(input_file)


    print(epsilon)
    print(C)
    private_agg_data = agg_data(original_data, epsilon, C)


    private_agg_data.to_csv(output_file, sep = ',')
 

if __name__ == "__main__":
    main()
