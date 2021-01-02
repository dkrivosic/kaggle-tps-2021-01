from sklearn import linear_model
import xgboost as xgb


models = {
    'sgd_regressor': linear_model.SGDRegressor(loss='squared_loss', max_iter=1000),
    'xgboost': xgb.XGBRegressor(n_jobs=-1, eta=0.0526333373190504, gamma=0.5786940216570573, max_depth=14,
                                min_child_weight=7, subsample=0.8532510562238894,
                                colsample_bytree=0.6831493557860527, reg_lambda=0.30521713570200254,
                                reg_alpha=0.7884678344268969)
}