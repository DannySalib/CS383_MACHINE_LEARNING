# Danny Salib
# CS383 Lab 5

import pandas as pd 
import numpy as np

def main():
    df = pd.read_table('spambase.data', header=None, delimiter=',')
    df = df.sample(frac=1)

    class_priors = df.iloc[:, -1]
    print('Class Priors')
    print(f'P(X=1) = {sum((class_priors == 1).astype(int)) / len(class_priors) * 100:.2f}%')
    print(f'P(X=0) = {sum((class_priors == 0).astype(int)) / len(class_priors) * 100:.2f}%')

    split = len(df) * 2 // 3

    X_tr = df.iloc[:split, :-1].to_numpy()
    X_v = df.iloc[split:, :-1].to_numpy()

    Y_tr = df.iloc[:split, -1].to_numpy()
    Y_v = df.iloc[split:, -1].to_numpy()

    ones = np.ones((X_tr.shape[0], 1))
    diagY_tr = np.diag(Y_tr)
    A = np.linalg.pinv(diagY_tr @ X_tr @ X_tr.T @ diagY_tr) @ ones

    W = X_tr.T @ diagY_tr @ A

    Yhat_tr = np.sign(X_tr @ W)
    Yhat_v = np.sign(X_v @ W)
    
    # Print ERROR%
    print(f'Training Accuracy (%): {accuracy(Yhat_tr, Y_tr) * 100:.2f}')
    print(f'Validating Accuracy (%): {accuracy(Yhat_v, Y_v) * 100:.2f}')
    print()
    print(f'Training Precision (%): {precision(Yhat_tr, Y_tr) * 100:.2f}')
    print(f'Validating Precision (%): {precision(Yhat_v, Y_v) * 100:.2f}')
    print()
    print(f'Training Recall (%): {recall(Yhat_tr, Y_tr) * 100:.2f}')
    print(f'Validating Recall (%): {recall(Yhat_v, Y_v) * 100:.2f}')
    print()
    print(f'Training F Measure (%): {f1_score(Yhat_tr, Y_tr) * 100:.2f}')
    print(f'Validating F Measure (%): {f1_score(Yhat_v, Y_v) * 100:.2f}')


accuracy = lambda Yhat, Y: (Yhat == Y).mean()

precision = lambda Yhat, Y: TP(Yhat, Y) / (TP(Yhat, Y) + FP(Yhat, Y)) if (TP(Yhat, Y) + FP(Yhat, Y)) != 0 else 0

recall = lambda Yhat, Y: TP(Yhat, Y) / (TP(Yhat, Y) + FN(Yhat, Y)) if (TP(Yhat, Y) + FN(Yhat, Y)) != 0 else 0

f1_score = lambda Yhat, Y: (2 * precision(Yhat, Y) * recall(Yhat, Y)) \
    / (precision(Yhat, Y) + recall(Yhat, Y)) if (precision(Yhat, Y) + recall(Yhat, Y)) != 0 else 0

# Compute confusion matrix for precision, recall, and harmonic mean
TP = lambda Yhat, Y: np.sum((Yhat == 1) & (Y == 1))  # True Positives
FP = lambda Yhat, Y: np.sum((Yhat == 1) & (Y != 1))  # False Positives
FN = lambda Yhat, Y: np.sum((Yhat != 1) & (Y == 1))  # False Negatives

if __name__ == '__main__':
    print('Running...')
    main()