import random
import numpy as np

class Perceptron:
    def __init__(self, type):
        if type == "digit":
            self.weights = np.random.rand(10, 784)
            self.bias = np.random.rand(10)
            self.type = "digit"
        else:
            self.weights = np.random.rand(4200) # change this later
            self.bias = random.random()
            self.type = "face"
    
    def get_weights(self):
        return self.weights

    def set_weight(self, new_weight, i):
        self.weights[i] = new_weight

    def get_bias(self):
        return self.bias

    def set_bias(self, bias):
        self.bias = bias


    ## DIGITS ##

    # p = perceptron, data = list of feature vectors
    def train_digits(self, percent_data, vectors, labels):
        #correct = 0
        # iterate through training data!
        for i in range (int(percent_data*1000)):
            
            vector = vectors[i]
            label = labels[i]
            #print("iteration " + str(i))
            #print(label)

            prediction_vector = np.matmul(self.weights, vector) + np.sum(self.bias) # for f_vector in feature_vectors
            pred_digit = np.argmax(prediction_vector)
            #print(pred_digit)
            #print(prediction_vector)

            # if correct, move on the next vector
            if pred_digit == label:
                #correct += 1
                continue
            else: # else, edit weights (rotate "plane" a bit towards the answer)
                self.weights[label] = self.weights[label] + vector # y
                self.weights[pred_digit] = self.weights[pred_digit] - vector # y'
                self.bias[label] = self.bias[label] + 1
                self.bias[pred_digit] = self.bias[pred_digit] - 1
                # can possibly improve by lowering the weights of other digits that are not correct, even though
                # they were not guessed

        # print("correct rate:")
        # print(correct/(percent_data*1000))

    def test_digits(self, vectors, labels):
        correct = 0
        for vector, label in zip(vectors, labels):
            prediction_vector = np.matmul(self.weights, vector) + np.sum(self.bias)
            if np.argmax(prediction_vector) == label:
                correct += 1
        print("Accuracy: " + str(correct/1000))

    ## FACES ##

    def train_faces(self, percent_data, vectors, labels):

        for i in range (int(percent_data*150)):
            
            vector = vectors[i]
            label = labels[i]

            prediction = np.dot(self.weights, vector) + self.bias

            #print(prediction)

            if (prediction > 0 and label == 1) or (prediction < 0 and label == 0): #if the prediction is correct
                #print("correct!")
                continue
            else:
                if prediction > 0: #if we predicted that it was a face when it wasn't
                    self.weights = self.weights - vector
                    self.bias = self.bias - 1
                else: #if we predicted that it wasn't a face when it was
                    self.weights = self.weights + vector
                    self.bias = self.bias + 1

    def test_faces(self, vectors, labels):
        correct = 0
        for vector, label in zip(vectors, labels):
            prediction = np.dot(self.weights, vector) + self.bias
            if (prediction > 0 and label == 1) or (prediction < 0 and label == 0): #if the prediction is correct
                correct += 1
        print("Accuracy: " + str(correct/150))
                