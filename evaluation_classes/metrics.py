from transformers import pipeline
import evaluate
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix, f1_score, precision_score, recall_score
import numpy as np


class Metrics:
    def __init__(self):
        pass

    def f1(self, predicted, actual):
        # metric = evaluate.load("f1")
        # return metric.compute(predictions = predicted, references = actual, average = "macro")
        print('F1 score: ', f1_score(predicted, actual, average = "macro")) 

    def precision(self, predicted, actual):
        # metric = evaluate.load("precision")
        # return metric.compute(predictions = predicted, references = actual, average = "macro")
        print('Precision: ', precision_score(predicted, actual, average = "macro")) 

    def recall(self, predicted, actual):
        # metric = evaluate.load("recall")
        # return metric.compute(predictions = predicted, references = actual, average = "macro")
        print('Recall: ', recall_score(predicted, actual, average = "macro"))

    def percent_within_range(self, predicted_scores, actual_scores, percentage):
        total_count = len(predicted_scores)
        count_within_range = 0
    
        for i in range(len(predicted_scores)):
            lower = actual_scores[i] - actual_scores[i]*percentage
            upper = actual_scores[i] + actual_scores[i]*percentage

            if lower <= predicted_scores[i] <= upper:
                count_within_range += 1
            
            return count_within_range/total_count
    
    def evaluate_scores(self, predicted_scores, actual_scores):
        percent_10_score = self.percent_within_range(predicted_scores, actual_scores, 0.1)
        percent_20_score = self.percent_within_range(predicted_scores, actual_scores, 0.2)
        percent_30_score = self.percent_within_range(predicted_scores, actual_scores, 0.3)
        print('Within 10 percent: ', percent_10_score)
        print('Within 20 percent: ', percent_20_score)
        print('Within 30 percent: ', percent_30_score)
    
    def mse(self, predicted_scores, actual_scores):
        predicted_scores = np.array(predicted_scores)
        actual_scores = np.array(actual_scores)
        print('Mean Squared Error: ', mean_squared_error(predicted_scores, actual_scores))
    
    def mae(self, predicted_scores, actual_scores):
        predicted_scores = np.array(predicted_scores)
        actual_scores = np.array(actual_scores)
        print('Mean Absolute Error: ', mean_absolute_error(predicted_scores, actual_scores))
    
    def conf_matrix(self, predicted, actual):
        matrix = confusion_matrix(predicted, actual)
        print(matrix)
    

            