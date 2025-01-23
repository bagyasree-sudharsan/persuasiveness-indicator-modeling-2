from baseline_classifier import baseline_classifier
from symbolic_classifier import symbolic_classifier
from baseline_regressor import baseline_regressor
from symbolic_regressor import symbolic_regressor

# # --------------------- Classifiers ----------------------------------------
# baseline_classifier('datasets/CMV_train.json', 0.85, 'baseline_cmv_classifier', 'BaselineCMVClassifier')
# baseline_classifier('datasets/SCOA_train.json', 0.85, 'baseline_scoa_classifier', 'BaselineSCOAClassifier')
symbolic_classifier('datasets/CMV_train_tagged.json', 'arg_comps_cmv_classifier', 'CMVClassifierArgComps', split_ratio = 0.85, use_sem_types = False) 
symbolic_classifier('datasets/CMV_train_tagged.json', 'sem_types_cmv_classifier', 'CMVClassifierSemTypes', split_ratio = 0.85, use_sem_types = True) 
symbolic_classifier('datasets/SCOA_train_tagged.json', 'arg_comps_scoa_classifier', 'SCOAClassifierArgComps', split_ratio = 0.85, use_sem_types = False) 
symbolic_classifier('datasets/SCOA_train_tagged.json', 'sem_types_scoa_classifier', 'SCOAClassifierSemTypes', split_ratio = 0.85, use_sem_types = True) 

# -------------------- Regression models -----------------------------------
# baseline_regressor('datasets/CMV_train.json', 0.95, 'baseline_cmv_regressor', 'BaselineCMVRegressor')
# baseline_regressor('datasets/SCOA_train.json', 0.95, 'baseline_scoa_regressor', 'BaselineSCOARegressor')
symbolic_regressor('datasets/CMV_train_tagged.json', 'arg_comps_cmv_regressor', 'CMVRegressorArgComps', use_sem_types = False) 
symbolic_regressor('datasets/CMV_train_tagged.json', 'sem_types_cmv_regressor', 'CMVRegressorSemTypes', use_sem_types = True) 
symbolic_regressor('datasets/SCOA_train_tagged.json', 'arg_comps_scoa_regressor', 'SCOARegressorArgComps', use_sem_types = False) 
symbolic_regressor('datasets/SCOA_train_tagged.json', 'sem_types_scoa_regressor', 'SCOARegressorSemTypes', use_sem_types = True) 

# # -------------------- Combined data -------------------------------------
# baseline_regressor('datasets/CMV_SCOA15.json', 0.95, 'baseline_cmv_scoa15_regressor', 'BaselineCMVSCOA15Regressor')
# baseline_regressor('datasets/CMV_SCOA30.json', 0.95, 'baseline_cmv_scoa30_regressor', 'BaselineCMVSCOA30Regressor')
# baseline_regressor('datasets/SCOA_CMV15.json', 0.95, 'baseline_scoa_cmv15_regressor', 'BaselineSCOACMV15Regressor')
# baseline_regressor('datasets/SCOA_CMV35.json', 0.95, 'baseline_scoa_cmv30_regressor', 'BaselineSCOACMV30Regressor')

# symbolic_regressor('datasets/CMV_SCOA15.json', 'sem_types_cmv_scoa15_regressor', 'CMVSCOA15Regressor', use_sem_types = True) 
# symbolic_regressor('datasets/CMV_SCOA30.json', 'sem_types_cmv_scoa30_regressor', 'CMVSCOA30Regressor', use_sem_types = True) 
# symbolic_regressor('datasets/SCOA_CMV15.json', 'sem_types_scoa_cmv15_regressor', 'SCOACMV15Regressor', use_sem_types = True) 
# symbolic_regressor('datasets/SCOA_CMV30.json', 'sem_types_scoa_cmv30_regressor', 'SCOACMV30Regressor', use_sem_types = True) 
