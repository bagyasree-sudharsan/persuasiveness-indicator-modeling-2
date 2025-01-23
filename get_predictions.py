from evaluation_classes.predictions import Prediction

prediction = Prediction()

model_path = 'models/TextSegArgComps'
dataset_path = 'datasets/processed/CMV/final.json'
texts = prediction.prepare_arg_comp_data_for_prediction(dataset_path)