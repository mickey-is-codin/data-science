import numpy as np
import pandas as pd

import tensorflow as tf

def main():

    print("Beginning program execution...")

    data_path = "data/jena_climate_2009_2016.csv"

    weather_df = pd.read_csv(data_path)

    print(weather_df.head())
    for column in list(weather_df.columns):
        print(column)

if __name__ == "__main__":
    main()
