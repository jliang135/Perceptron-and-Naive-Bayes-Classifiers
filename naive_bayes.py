import math
import numpy as np
import random

class NB_mnist:
    def __init__(self, type, vector_len): #vector_len = length of feature vectors
        self.type = type
        self.vector_len = vector_len
        self.digits_count = [0] * 10
        self.distribution = [0] * 10 # change for faces
        self.cond_probs = np.zeros((10, 784))

    # NOTE: you already need to have selected the images/vectors you are planning on training with
    def get_occurrences(self, vectors, labels):
        n = len(vectors)
        #print("training size:")
        #print(n)
        for label in labels:
            self.digits_count[label] += 1
        
        for i in range(len(self.digits_count)):
            self.distribution[i] = self.digits_count[i]/n
        
        #print(self.digits_count)
        #print(self.distribution)

    def pixel_prob(self, vectors, labels):
        counts = []
        for i in range(10):
            counts.append(np.zeros(784))

        for i in range(len(vectors)):
            counts[labels[i]] = counts[labels[i]] + vectors[i]

        #print(counts)     

        for i in range(10):
            for j in range(self.vector_len):
                cond_prob = (counts[i][j]+0.001)/(self.digits_count[i]+0.001) # + 0.001 so it's never 0, can't take the log of 0
                self.cond_probs[i][j] = cond_prob 

        # same = True
        # for x, y in zip(self.cond_probs[0], self.cond_probs[3]):
        #     if x != y: same = False
        # print(same)

        # print(self.cond_probs)
        # for c in self.cond_probs:
        #     print(c)

    def predict_mnist(self, vector):
        count = [0] * 10
        for j in range(self.vector_len):
            if vector[j] == 1:
                for i in range(len(count)): # add log to every value in count
                    count[i] = count[i] + math.log(self.cond_probs[i][j]) # conditional probabilities
            else:
                for i in range(len(count)):
                    if self.cond_probs[i][j] == 1:
                        self.cond_probs[i][j] -= 0.1
                for i in range(len(count)):
                    # every value in cond_probs must be < 1 bc its probability bro
                    count[i] += math.log(1-self.cond_probs[i][j])
        
        # probability that this vector is the digit at index i
        # print(count)
        prediction_vector = np.zeros(10)
        for i in range(prediction_vector.size):
            prediction_vector[i] = count[i] + math.log(self.distribution[i])

        return np.argmax(prediction_vector)

    # trains
    def training_mnist(self, percent_data, all_vectors, all_labels):

        n = int(percent_data*len(all_vectors))

        # only using some data -- based on given percentage
        # randomize
        vectors = []
        labels = []
        i_list = []
        for i in range(n):
            index = random.randint(0, len(all_vectors)-1)
            while index in i_list:
                index = random.randint(0, len(all_vectors)-1)
            i_list.append(index)
            vectors.append(all_vectors[index])
            labels.append(all_labels[index])

        # print(labels)

        # print(len(vectors))

        self.get_occurrences(vectors, labels)
        self.pixel_prob(vectors, labels)

        # print(self.distribution)

        #correct = 0
        for i in range(len(vectors)):
            # print(i, end = ": ")
            pred_digit = self.predict_mnist(vectors[i])
            # print(pred_digit, end=", ")
            # print(labels[i])
            #if pred_digit == labels[i]:
                #correct += 1
        #print(correct/len(vectors))

    def testing_mnist(self, vectors, labels):
        correct = 0
        for i in range(len(vectors)):
            # print(i, end = ": ")
            pred_digit = self.predict_mnist(vectors[i])
            # print(pred_digit, end=", ")
            # print(labels[i])
            if pred_digit == labels[i]:
                correct += 1
        print(correct/len(vectors))







## FACES ##

class NB_faces:
    def __init__(self, type, vector_len): #vector_len = length of feature vectors
        self.type = type
        self.vector_len = vector_len
        self.faces_count = 0
        self.non_faces_count = 0
        self.count = [0, 0] #[non-face, face]
        self.distribution = [0, 0] #[non-face, face]
        self.cond_probs = np.zeros((2, 4200))

    # NOTE: you already need to have selected the images/vectors you are planning on training with
    def get_occurrences(self, vectors, labels):
        n = len(vectors)
        #print("training size:")
        #print(n)
        
        for label in labels:
            if label == 1:
                self.faces_count += 1
            else:
                self.non_faces_count += 1

        self.count[0] = self.non_faces_count
        self.count[1] = self.faces_count
        
        self.distribution[0] = self.non_faces_count/n
        self.distribution[1] = self.faces_count/n

    def pixel_prob(self, vectors, labels):
        counts = []
        
        for i in range(2):
            counts.append(np.zeros(4200)) #counts[0] is no face, counts[1] is face

        for i in range(len(vectors)):
            counts[labels[i]] = counts[labels[i]] + vectors[i]

        #print(counts)     

        for i in range(2):
            for j in range(self.vector_len):
                cond_prob = (counts[i][j]+0.001)/(self.count[i]+0.001) # + 0.001 so it's never 0, can't take the log of 0
                self.cond_probs[i][j] = cond_prob 

        # same = True
        # for x, y in zip(self.cond_probs[0], self.cond_probs[3]):
        #     if x != y: same = False
        # print(same)

        # print(self.cond_probs)
        # for c in self.cond_probs:
        #     print(c)

    def predict_faces(self, vector):
        count = [0] * 2
        for j in range(self.vector_len):
            if vector[j] == 1:
                for i in range(len(count)): # add log to every value in count
                    count[i] = count[i] + math.log(self.cond_probs[i][j]) # conditional probabilities
            else:
                for i in range(len(count)):
                    if self.cond_probs[i][j] == 1:
                        self.cond_probs[i][j] -= 0.1
                for i in range(len(count)):
                    # every value in cond_probs must be < 1 bc its probability bro
                    count[i] += math.log(1-self.cond_probs[i][j])
        
        # probability that this vector is the digit at index i
        # print(count)
        prediction_vector = np.zeros(2)
        for i in range(prediction_vector.size):
            prediction_vector[i] = count[i] + math.log(self.distribution[i])

        return np.argmax(prediction_vector)

    # trains
    def training_faces(self, percent_data, all_vectors, all_labels):

        n = int(percent_data*len(all_vectors))

        # only using some data -- based on given percentage
        # randomize
        vectors = []
        labels = []
        i_list = []
        for i in range(n):
            index = random.randint(0, len(all_vectors)-1)
            while index in i_list:
                index = random.randint(0, len(all_vectors)-1)
            i_list.append(index)
            vectors.append(all_vectors[index])
            labels.append(all_labels[index])

        # print(labels)

        # print(len(vectors))

        self.get_occurrences(vectors, labels)
        self.pixel_prob(vectors, labels)

        # print(self.distribution)

        #correct = 0
        for i in range(len(vectors)):
            # print(i, end = ": ")
            pred_digit = self.predict_faces(vectors[i])
            #print(pred_digit, end=", ")
            #print(labels[i])
            #if pred_digit == labels[i]:
                #correct += 1
        #print(correct/len(vectors))

    def testing_faces(self, vectors, labels):
        correct = 0
        for i in range(len(vectors)):
            # print(i, end = ": ")
            pred_digit = self.predict_faces(vectors[i])
            # print(pred_digit, end=", ")
            # print(labels[i])
            if pred_digit == labels[i]:
                correct += 1
        print(correct/len(vectors))



        

