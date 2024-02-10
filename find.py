 # Sentiment Analysis of Movie Reviews 




# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing deep learning libraries
import tensorflow as tf
from tensorflow import keras
from keras import Input
from keras.models import Sequential
from keras.layers import Dense, Flatten
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer



## Loading dataset

# Setting seed
seed = 12345

test_size = 1000
train_size = 4000

train_list, test_list = keras.datasets.imdb.load_data(seed = seed)
word_dict = keras.datasets.imdb.get_word_index()


# Split to create validation set
x_val_list, x_test_list, y_val, y_test = train_test_split(test_list[0][:test_size*2],
                                                          test_list[1][:test_size*2], test_size = .5, random_state = seed, shuffle = True)

print(f'Training Data Shape: {train_list[0].shape}')
print(f'Validation Data Shape: {x_val_list.shape}')
print(f'Testing Data Shape: {x_test_list.shape}')

# Turning list of numbers into strings for the vectorizer
train_string = [' '.join(map(str, review)) for review in train_list[0][:train_size]]
test_string = [' '.join(map(str, review)) for review in x_test_list]
val_string = [' '.join(map(str, review)) for review in x_val_list]

# Creating a Vectorizer to convert our data into Bag if Words Representations based on word count
vectorizer = CountVectorizer()
vectorizer.fit(train_string)

x_train = vectorizer.transform(train_string)
x_test = vectorizer.transform(test_string)
x_val = vectorizer.transform(val_string)

y_train = train_list[1][:train_size]


# defining the structure of the model 
model = Sequential()
model.add(Dense(128,activation='relu', input_dim=x_train.shape[1])) # These Dense layers are our neuron layers
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid')) # Sigmoid scales our output between 0 and 1 for classifying 'negative' or 'positive'


# Compiling the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the model
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val, y_val))

# Evaluating performance of the model on the test data
model.evaluate(x=x_test, y=y_test, batch_size = 32)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Testing the model with my own reviews 
def input_to_encoding(input: str, word_dict):

    # Convert input string to lowercase and remove non-alphanumeric characters
    cleaned_string = ''.join(char.lower() if char.isalnum() or char.isspace() else ' ' for char in input)

    # Split the cleaned string into a list of words
    words = cleaned_string.split()

    # Map each word to its corresponding integer value using the provided dictionary
    encoding = [str(word_dict.get(word, 0)) for word in words]

    encoding_string = [' '.join(encoding)]

    return encoding_string

review = "That was not as bad as I thought it would be, but it nearly was"

# Encode our own review
encoding = input_to_encoding(review, word_dict)

# Transform it using the vectorizer
input = vectorizer.transform(encoding)

# Feed our encoded review into the model to predict the sentiment
prediction = model.predict(input)

print("Model Prediction: ", 'Positive' if prediction==1 else 'Negative')