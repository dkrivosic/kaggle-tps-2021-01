from sklearn import linear_model
import xgboost as xgb


models = {
    'sgd_regressor': linear_model.SGDRegressor(loss='squared_loss', max_iter=1000),
    'xgboost': xgb.XGBRegressor(n_jobs=-1)
}