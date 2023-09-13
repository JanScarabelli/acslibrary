from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error

import numpy as np


def r2_adjusted(y_true, y_pred, x_dim):
    return 1 - (1 - r2_score(y_true, y_pred)) * (len(y_true) - 1) / (len(y_true) - x_dim - 1)


def rmsle(y_test, y_pred):
    y_pred[y_pred < 0] = 0
    return np.sqrt(mean_squared_log_error(y_test, y_pred))


def rmse(y_test, y_pred):
    y_pred[y_pred < 0] = 0
    return np.sqrt(mean_squared_error(y_test, y_pred))
