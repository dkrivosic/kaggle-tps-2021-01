import argparse
import pickle

import pandas as pd

import config


def run(model):
    with open(model, 'rb') as f:
        reg_model = pickle.load(f)

    df = pd.read_csv(config.TEST_FILE)
    x_test = df.drop('id', axis=1).values
    y_pred = reg_model.predict(x_test)
    
    df['target'] = y_pred
    feature_columns = [column for column in df.columns
                       if column not in ['id', 'target']]
    df = df.drop(feature_columns, axis=1)
    df.to_csv(config.SUBMISSION_FILE, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()

    run(args.model)
