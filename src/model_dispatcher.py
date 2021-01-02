from sklearn import linear_model


models = {
    'sgd_regressor': linear_model.SGDRegressor(loss='squared_loss', max_iter=1000)
}