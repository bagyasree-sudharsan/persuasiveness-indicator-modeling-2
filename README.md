All relevant code is in final/.

1. data_subsets.py generates random train and test subsets for the various datasets to run experiments on. 
2. arg_comp_classifier.py trains ArgCompClassifier on AnnotatedCMV, and sem_type_classifier trains SemTypeClassifier on AnnotatedCMV. AnnotatedCMV is the only dataset with human-annotated ground truth values for these. 
3. ArgCompClassifier and SemTypeClassifier are used in predict_tags to predict the argument components and semantic types for the subsets created in data_subsets.py. These are used as ground truth values for these datasets. 
4. train_models.py trains the different classifiers and regressors for the experiments. Code can be commented/uncommented based on which models need to be trained. 
5. Actual training code is present in baseline_classifier.py, baseline_regressor.py, symbolic_classifier.py, symbolic_regressor.py.
6. evaluate_models.py evaluates the models. metrics.py contains the metrics used for evaluation. 
7. common.py has custom modules used across various files. 