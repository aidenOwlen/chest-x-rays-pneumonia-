import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import tensorflow as tf 
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential 
from  tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout
from sklearn.metrics import classification_report, confusion_matrix
import os 

def plot_pie_normal_vs_positive():
	"""plot pie figure to analyze the class imbalance"""
    plt.figure(figsize=(20,30))
    plt.pie([len(pneumonia_train),len(normal_train)], labels = ["Pneumonie", "Normal"])
    plt.title("Pneumonie vs Normal")
    plt.show()


#Directories of images 
train_dir = "chest_xray/train/"
test_dir = "chest_xray/test/"
val_dir = "chest_xray/val/"

#images shape
image_size = 220
input_shape = (220,220,3)

#Path to uimages
pneumonia_train = [img for img in os.listdir(train_dir + "PNEUMONIA/")]
normal_train = [img for img in os.listdir(train_dir + "NORMAL/")]
#print(len(normal_train))
#print(len(pneumonia_train))

data_generator = ImageDataGenerator(rescale = 1./255, zoom_range = 0.2)
train_generator = data_generator.flow_from_directory(train_dir,target_size = (image_size,image_size),class_mode = "binary", batch_size=20)
validation_generator = data_generator.flow_from_directory(val_dir,target_size = (image_size,image_size),class_mode = "binary", batch_size=20)
test_generator = data_generator.flow_from_directory(test_dir,target_size = (image_size,image_size),class_mode = "binary", batch_size=20)


base_model = Sequential()
base_model.add(tf.keras.applications.ResNet50V2(include_top = False,input_shape = input_shape, weights = "imagenet"))
for layer in base_model.layers:
    layer.trainable = False

model = Sequential() #Create CNN
model.add(base_model)  #Add resnev50v2 to our neural network
model.add(GlobalAveragePooling2D()) #apply averagepooling2D
model.add(Dense(128, activation = "relu")) #Add layer with 128 trainable nodes, activation = relu
model.add(Dropout(0.2)) #add dropout keep_prob>0.8
model.add(Dense(1,activation = "sigmoid")) #Ouput layer sigmoid ( btw 0 and 1) : binary

print(model.summary()) #Print architecture of our model

#compile our model using accuracy as metrics, adam for back_ward prop optimization, loss binary crossentropy
model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"]) 

#Add earlystopping based on accuracy : meaning when it just starts to overtfit : quit
#patience : nb of epochs with no improvement after which we stop the training
callback = tf.keras.callbacks.EarlyStopping(monitor="accuracy", patience=4)

#Train the model and capture history of loss and accuracy
history = model.fit(train_generator,validation_data=validation_generator, steps_per_epoch = 100, epochs = 20, callbacks = callback)

#Save model
model.save("model_xray")

#Print metrics
accuracy = history.history['accuracy']
val_accuracy  = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
print(accuracy)
print(val_accuracy)



