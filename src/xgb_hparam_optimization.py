from functools import partial

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from skopt import space, gp_minimize

import config


def optimize(params, param_names, x, y):
    params = dict(zip(param_names, params))
    model = xgb.XGBRegressor(**params, n_jobs=-1)
    kf = KFold(n_splits=5, shuffle=True, random_state=config.KFOLD_SEED)

    errors = []
    for train_idx, val_idx in kf.split(X=x, y=y):
        x_train = x[train_idx]
        y_train = y[train_idx]
        x_val = x[val_idx]
        y_val = y[val_idx]

        model.fit(x_train, y_train)
        y_pred = model.predict(x_val)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        errors.append(rmse)
    return np.mean(errors)
        

if __name__ == '__main__':
    df = pd.read_csv('input/train.csv')
    df = df.drop('id', axis=1)
    x = df.drop('target', axis=1).values
    y = df.target.values

    param_space = [
        space.Real(0.01, 0.1, name='eta'),
        space.Real(0.05, 1.0, name='gamma'),
        space.Integer(3, 25, name ='max_depth'),
        space.Integer(1, 7, name='min_child_weight'),
        space.Real(0.6, 1.0, name='subsample'),
        space.Real(0.6, 1.0, name='colsample_bytree'),
        space.Real(0.01, 1.0, name='lambda'),
        space.Real(0.0, 1.0, name='alpha')
    ]

    param_names = ['eta', 'gamma', 'max_depth', 'min_child_weight',
                   'subsample', 'colsample_bytree', 'lambda', 'alpha']
    
    optimization_function = partial(
        optimize,
        param_names=param_names,
        x=x,
        y=y
    )

    result = gp_minimize(
        optimization_function,
        dimensions=param_space,
        n_calls=15,
        n_random_starts=10,
        verbose=10
        )
    
    best_params = dict(zip(param_names, result.x))
    print(best_params)
