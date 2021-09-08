import numpy as np
from perceptron import Perceptron
from naive_bayes import NB_mnist, NB_faces
import time

class all_data:
    def __init__(self, images, vectors, labels):
        self.images = images
        self.labels = labels
        self.vectors = vectors

def get_features(image):
    pixel_list = image.flatten()
    vector = []
    for pixel in pixel_list:
        if pixel == " ":
            vector.append(0)
        elif pixel == "\n":
            continue
        else:
            vector.append(1)
    return np.array(vector)

# create 1000 length array for 1000 images
# digits are 28x28, 1000
# faces are 70x60, 150
def load_images(file_name, labels_name, type):

    f1 = open(file_name, 'r')
    image_lines = f1.readlines()

    f2 = open(labels_name, 'r')
    label_lines = f2.readlines()

    images = []

    if type == "digit":
        h = w = 28
        total = 1000
    else:
        h = 70
        w = 60
        total = 150

    images = []
    vectors = []
    labels = []

    # get list of images, list of vectors, list of labels
    for i in range(0, total): #just 10 for now, use total for this later
        image = []
        for j in range(i*h, (i+1)*h):
            image.append(list(image_lines[j]))
        image = np.array(image)
        vectors.append(get_features(image))
        labels.append(int(label_lines[i][0]))
        images.append(image)

    return all_data(images, vectors, labels)

    # for vector in vectors:
    #     for i in range(0, 784, 28):
    #         for j in range(28):
    #             print(vector[i+j], end="")
    #         print()

    # for vector in vectors:
    #     for i in range(0, 4200, 60):
    #         for j in range(60):
    #             print(vector[i+j], end="")
    #         print()

    # for image in images:
    #     image.print_image()
    
    #print(get_features(images[0]))
    
# LOADING TRAINING AND TESTING IMAGES AND LABELS FOR MNIST AND FACES
mnist_training_data = load_images('data/digitdata/trainingimages', 'data/digitdata/traininglabels', 'digit')
mnist_testing_data = load_images('data/digitdata/testimages', 'data/digitdata/testlabels', 'digit')

face_training_data = load_images('data/facedata/facedatatrain', 'data/facedata/facedatatrainlabels', 'face')
face_testing_data = load_images('data/facedata/facedatatest', 'data/facedata/facedatatestlabels', 'face')

## --- PERCEPTRON ---

percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

print("\n\n PERCEPTION - MNIST \n\n")

for p in percentages:
    print("Training MNIST Perceptron with " + str(p*100) + " percent of the MNIST data... \n")
    mnist_p = Perceptron("digit")
    start = time.time()
    mnist_p.train_digits(p, mnist_training_data.vectors, mnist_training_data.labels)
    end = time.time()
    print(" > Training Time: " + str(end-start))
    print()
    print("Testing MNIST Perceptron: \n")
    print(" > Accuracy: ", end="")
    mnist_p.test_digits(mnist_testing_data.vectors, mnist_testing_data.labels)
    print("-- -- -- -- --")
    print()

print("\n\n PERCEPTION - FACES \n\n")
   
for p in percentages:
    print("Training FACE Perceptron with " + str(p*100) + " percent of the FACE data... \n")
    face_p = Perceptron("face")
    start = time.time()
    face_p.train_faces(p, face_training_data.vectors, face_training_data.labels)
    end = time.time()
    print(" > Training Time: " + str(end-start))
    print()
    print("Testing MNIST Perceptron: \n")
    print(" > Accuracy: ", end="")
    face_p.test_faces(face_testing_data.vectors, face_testing_data.labels)
    print("-- -- -- -- --")
    print()

## --- NAIVE BAYES ---

print("\n\n NAIVE BAYES - MNIST \n\n")

for p in percentages:
    print("Training MNIST Naive-Bayes with " + str(p*100) + " percent of the MNIST data... \n")
    mnist_nb = NB_mnist("digit", 784)
    start = time.time()
    mnist_nb.training_mnist(p, mnist_training_data.vectors, mnist_training_data.labels)
    end = time.time()
    print(" > Training Time: " + str(end-start))
    print()
    print("Testing MNIST Naive-Bayes: \n")
    print()
    print("Accuracy: ", end="")
    mnist_nb.testing_mnist(mnist_testing_data.vectors, mnist_testing_data.labels)
    print("-- -- -- -- --")
    print()

print("\n\n NAIVE BAYES - FACES \n\n")

for p in percentages:
    print("Training FACE Naive-Bayes with " + str(p*100) + " percent of the FACE data... \n")
    faces_nb = NB_faces("faces", 4200)
    start = time.time()
    faces_nb.training_faces(p, face_training_data.vectors, face_training_data.labels)
    end = time.time()
    print(" > Training Time: " + str(end-start))
    print()
    print("Testing FACE Naive-Bayes: \n")
    print()
    print("Accuracy: ", end="")
    faces_nb.testing_faces(face_testing_data.vectors, face_testing_data.labels)
    print("-- -- -- -- --")
    print()