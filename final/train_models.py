from baseline_regressor import baseline_regressor
from symbolic_regressor import symbolic_regressor

# baseline_regressor('datasets/CMV_train.json', 0.95, 'baseline_cmv_regressor', 'BaselineCMVRegressor')
baseline_regressor('datasets/SCOA_train.json', 0.95, 'baseline_scoa_regressor', 'BaselineSCOARegressor')