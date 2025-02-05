# Danny Salib
# CS383 ML
# Lab 4 - linear Regression 

import pandas as pd
import numpy as np

# Constants 
DATA_PATH = 'spambase.data'
EPOCHS = range(10000)
ETA = 0.1
EPSILON = 10**-7
DIFF_THRESHOLD = 2**-32

# Helper fucntions 
sigmoid = lambda X, W: 1 / (1 + np.exp(-np.dot(X, W)))
log_loss = lambda Yhat, Y: np.mean(
    -Y * np.log(Yhat + EPSILON) - (1 - Y) * np.log(1 - Y + 1e-10))
classify = lambda Y: (Y >= 0.5).astype(int)
get_target_values = lambda df: df.iloc[:, -1]
drop_target_values = lambda df: df.iloc[:, :-1]

def main():
    # Initialize
    np.random.seed(0)

    df = pd.read_table('spambase.data', delimiter=',', header=None)
    df = df.sample(frac=1) # Shuffle df

    # Split into training and validating dfs
    split_index = len(df) * 2 // 3
    df_tr = df.iloc[:split_index, :]
    df_v = df.iloc[split_index:, :]

    # Separate features from target
    Y_tr = get_target_values(df_tr).to_numpy()
    Y_v = get_target_values(df_v).to_numpy()

    X_tr = drop_target_values(df_tr).to_numpy()
    X_v = drop_target_values(df_v).to_numpy()

    # Standadrize data based on TRAINING data only 
    m = np.mean(X_tr, axis=0, keepdims=True)
    s = np.std(X_tr, axis=0, keepdims=True)
    zScore = lambda X: (X - m) / s

    X_tr = zScore(X_tr)
    X_v = zScore(X_v)

    # Insert bias features 
    X_tr = np.insert(X_tr, 0, 1, axis=1)
    X_v = np.insert(X_v, 0, 1, axis=1)

    # Initialize weights randomly in range [-10^-4, 10^-4]
    W = np.random.uniform(-1e-4, 1e-4, size=X_tr.shape[1])

    # Calculate gradient
    # Adjust Weights based on ETA, EPOCHS, and EPSILON
    prev_loss_tr = float('inf')
    losses_tr, losses_v = [], [] # Store log losses
    Yhat_tr: None
    Yhat_v: None

    for _ in EPOCHS:
        # Calculate log losses and store them
        # Validating 
        Yhat_v = sigmoid(X_v, W)
        loss_v = log_loss(Yhat_v, Y_v)
        losses_v.append(loss_v)
        # Training
        Yhat_tr = sigmoid(X_tr, W)
        loss_tr = log_loss(Yhat_tr, Y_tr)
        losses_tr.append(loss_tr)

        # Calculate gradiant 
        g = (1 / X_tr.shape[0]) * np.dot(X_tr.T, (Yhat_tr - Y_tr))
        W -= (ETA * g)  # update weight
        
        if abs(prev_loss_tr - loss_tr) < DIFF_THRESHOLD:
            break

        prev_loss_tr = loss_tr

    # Classify our results 
    Yhat_tr = sigmoid(X_tr, W)
    Yhat_v = sigmoid(X_v, W)

    Yhat_tr = classify(Yhat_tr)
    Yhat_v = classify(Yhat_v)


if __name__ == '__main__':
    main()
 