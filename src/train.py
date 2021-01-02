import os
import argparse
import time
from datetime import timedelta
import pickle

import pandas as pd
from sklearn.metrics import mean_squared_error

import config
import model_dispatcher



def run(model, fold):
    df = pd.read_csv(config.TRAINING_FILE)

    df_train = df[df['kfold'] != fold]
    x_train = df_train.drop(['kfold', 'target'], axis=1).values
    y_train = df_train.target.values

    start_time = time.time()
    reg_model = model_dispatcher.models[model]
    reg_model.fit(x_train, y_train)
    training_time = time.time() - start_time
    training_time = str(timedelta(seconds=training_time))

    df_val = df[df['kfold'] == fold]
    x_val = df_val.drop(['kfold', 'target'], axis=1).values
    y_val = df_val.target.values
    y_pred = reg_model.predict(x_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)

    print('Fold: %d, RMSE: %.5f, Training time: %s' % (fold, rmse, training_time))

    model_filename = model + '_fold_' + str(fold) + '.pickle'
    model_path = os.path.join(config.MODEL_OUTPUT, model_filename)
    with open(model_path, 'wb') as f:
        pickle.dump(reg_model, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--folds', type=int, default=5)
    args = parser.parse_args()

    for fold in range(args.folds):
        run(args.model, fold)