
class Evaluation:
    def __init__():
        pass

    def binary_f1(self, predicted, actual):
        pass
    def binary_precision(self, predicted, actual):
        pass
    def binary_recall(self, predicted, actual):
        pass

    def evaluate_by_range(self, predicted_scores, actual_scores, percentage):
        predicted = []
        actual = []

        for i in range(len(predicted_scores)):
            actual_value = 0 if actual_scores[i] < 0.5 else 1
            actual.append(actual_value)
            
            lower = actual_scores[i] - actual_scores[i]*percentage
            upper = actual_scores[i] + actual_scores[i]*percentage
            
            predicted_value = actual_value if lower <= predicted_scores[i] <= upper else abs(actual_value - 1)
            predicted.append(predicted_value)

            #Check if this is the best metric. Otherwise, I can just find the percentage of scores that are within the range.
            return self.binary_f1(predicted, actual)
    
    def evaluate_scores(self, predicted_scores, actual_scores):
        percent_10_score = self.evaluate_by_range(predicted_scores, actual_scores, 0.1)
        percent_20_score = self.evaluate_by_range(predicted_scores, actual_scores, 0.2)
        percent_30_score = self.evaluate_by_range(predicted_scores, actual_scores, 0.3)
        return percent_10_score, percent_20_score, percent_30_score

            