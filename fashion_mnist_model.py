import tensorflow as tf
from tensorflow import keras
# Tensorflow and tf.keras

#helper libraries
import numpy as np
import matplotlib.pyplot as plt

#print("Hello World")

#Use MNIST Fasion Dataset, included in keras.
#includes 60,000 images for training and 10,000 images for validation/testing

fashion_mnist = keras.datasets.fashion_mnist # load dataset

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data() #split function into testing and training

#print(train_images.shape) #output will be (60000, 28, 28). We have 60000 images that are made up of 28x28 pixels (784 in total
#print(type(train_images)) #output = <class 'numpy.ndarray'>

#print(train_images[0,23,23]) #output will be 194, this is the grayscale value. Pixel values will be between 0 and 255, 0 being black and 255 being white.

#print(train_labels[:10]) #outputs array of integers ranging from 0 to 9, each integer represents a specific article of clothing. Create an array of label names below to indicate which is which.

class_names = ['T-short/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

#Use matplotlib to show what one of the images look like

# plt.figure() #make a figure
# plt.imshow(train_images[3]) #show the image
# plt.colorbar() #make colorbar
# plt.grid(False) # no grid
# plt.show() # show the image

#-------------Data Preprocessing------------#

"""
Last step before creating model is to preprocess our data. This means apply some prior transformations to our data before feeding it to the model. In this case, we will
simply scale all of our greyscale pixel values (0-255) to be between 0 and 1. We can do this by dividing each value in the training and testing sets by 225.0. We do this becayse smaller values 
will make it easier for the model to process our values
"""

train_images = train_images/255.0 #Preprocessing 
test_images = test_images/255.0 #Preprocessing

#-------------Making the model------------#

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), #input layer (1), Flatten takes all 724 (28x28) pixels from a matrix structure to a flat line of pixels
    keras.layers.Dense(128, activation="relu"), #hidden layer (2) Dense layer, all neurons in the previous layer is connected to this layer. 128 neurons
    keras.layers.Dense(10, activation="softmax") #output layer (3) 10 output neurons with acivation 'softmax'. Output layer must have as many neurons as classes you are. predicting for
    #softmax ensures all of the values of your neurons add up to one and that they are between 0 and 1.
])

#-------------Compiling the model------------#

model.compile(optimizer='adam', # algortithm that performs gradient descent
              loss = 'sparse_categorical_crossentropy', # loss function
              metrics=['accuracy']) #Pick different values for hyperparameter tuning

#-------------Training the model------------#
model.fit(train_images, train_labels, epochs=10) 

#-------------Evaluating the model------------#

#use built-in method from keras
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)
print('Test Accuracy: ', test_acc) #Test accuracy for first test was 87.783. Lower than training numbers because of  overfitting. Model is more accurate on data that has been repeatedly fed.
#feeding new data lowers accuracy. Want highest accuracy possible on new data. Be aware of overfeeding in training! Less epochs might be better

#-------------Make Predictions------------#
predictions = model.predict(test_images)
#print(class_names[np.argmax(predictions[5])]) # np.argmax returns the highest value in array of predictions

# plt.figure() #make a figure
# plt.imshow(train_images[5]) #show the image
# plt.colorbar() #make colorbar
# plt.grid(False) # no grid
# plt.show() # show the image

#-------------Verify Predictions------------#
# COLOR = 'white'
# plt.rcParams['text.color'] = COLOR
# plt.rcParams['axes.labelcolor'] = COLOR

# def predict(model, image, correct_label):
#     class_names = ['T-short/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
#     prediction = model.predict(np.array([image]))
#     predicted_class = class_names[np.argmax(prediction)]

#     show_image(image, class_names[correct_label], predicted_class)

# def show_image(img, label, guess):
#     plt.figure()
#     plt.imshow(img, cmap=plt.cm.binary)
#     plt.title("Expected: " + label)
#     plt.xlabel("Guess: " + guess)
#     plt.colorbar()
#     plt.grid(False)
#     plt.show()

# def get_number():
#     while True:
#         num = input("Pick a number: ")
#         if num.isdigit():
#             num = int(num)
#             if 0<= num <= 1000:
#                 return int(num)
#         else:
#             print("Try Again...")

# num=get_number()
# image = test_images[num]
# label = test_labels[num]
# predict(model, image, label)
# show_image(image, label, num)