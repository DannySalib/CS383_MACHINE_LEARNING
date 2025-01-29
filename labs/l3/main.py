# Danny Salib
# Jan 29 2025
# Lab 3 - Linear Regression 

import pandas as pd
import numpy as np

DATA_PATH = './insurance.csv'

def main():
    df = pd.read_csv(DATA_PATH)

    # one-hot-encode sex
    df['is_male'] = (df['sex'] == 'male').astype(int)
    df['is_female'] = (df['sex'] == 'female').astype(int)

    # convert smoker col to binary
    df['smoker'] = (df['smoker'] == 'yes').astype(int)
    
    # One-hot-encode region 
    regions = df['region'].unique()
    for region in regions:
        df[f'is_{region}'] = (df['region'] == region).astype(int)

    # Drop old cols
    df = df.drop(['sex', 'bmi', 'region'], axis=1)
    
    # Shuffle Data
    df = df.sample(frac=1)


if __name__ == '__main__':
    main()