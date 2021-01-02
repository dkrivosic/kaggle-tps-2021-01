import argparse

import pandas as pd
from sklearn.model_selection import KFold

import config


def run(filename, folds):
    df = pd.read_csv(filename)
    
    df['kfold'] = -1
    kf = KFold(n_splits=folds, shuffle=True, random_state=config.KFOLD_SEED)
    for fold, (_, val_) in enumerate(kf.split(X=df)):
        df.loc[val_, 'kfold'] = fold

    df = df.drop(['id'], axis=1)

    output_filename = filename.split('.')[0] + '_folds.csv'
    df.to_csv(output_filename, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True)
    parser.add_argument('--folds', type=int, default=5)
    args = parser.parse_args()

    run(args.file, args.folds)