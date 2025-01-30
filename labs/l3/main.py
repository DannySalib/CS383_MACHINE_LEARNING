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
    df.insert(0, 0, 1) # we need an extra feature for the weight intercept 

    # Split into training data (2/3) and test data (1/3)
    split_index = (len(df) * 2) // 3
    training_df = df.iloc[:split_index, :]
    validating_df = df.iloc[split_index:, :]

    # Target values
    Y_tr = training_df['charges'].to_numpy()
    Y_v = validating_df['charges'].to_numpy()
    X_tr = training_df.drop('charges', axis=1).to_numpy()
    X_v = validating_df.drop('charges', axis=1).to_numpy()

    # Calculate our weight 
    W = np.linalg.pinv(X_tr) @ Y_tr

    # Calculate results
    result = lambda X: W[0] + np.dot(X[:, 1:], W[1:])
    Yhat_tr = result(X_tr)
    Yhat_v = result(X_v)
    
    # Display results 
    print(f'Training RMSE: {RMSE(Y_tr, Yhat_tr)}')
    print(f'Validation RMSE: {RMSE(Y_v, Yhat_v)}')
    print(f'Training SMAPE (%): {SMAPE(Y_tr, Yhat_tr)}')
    print(f'Validation SMAPE (%): {SMAPE(Y_v, Yhat_v)}')

def RMSE(Y, Yhat):
    return np.sqrt(np.mean((Y - Yhat) ** 2))

def SMAPE(Y, Yhat):
   numerator = np.abs(Y - Yhat)
   denominator = (np.abs(Y) + np.abs(Yhat))
   return np.mean(numerator / denominator) * 100  # Convert to percentage

if __name__ == '__main__':
    main()