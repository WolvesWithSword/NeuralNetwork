import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
import PIL
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, regularizers
from tensorflow.python.keras.layers.normalization import BatchNormalization
from keras.models import load_model
from tensorflow.python.keras.applications.densenet import decode_predictions
from tensorflow.python.ops.metrics_impl import mean_per_class_accuracy
from report import *


img_dir = "./image/"
LABEL = ["buildings","forest","glacier","mountain","sea","street"]

def main():
    #autoTrain()
    models = loadModel("model/best.h5")
    models.summary()
    predict(models,"./image/seg_test/seg_test/glacier/20087.jpg",'probabilities')
    y_test,real = reportTestData(models)
    reportText(y_test,real,LABEL)
    #reportGraphe(y_test,real,LABEL)
    reportRepartition(y_test,real,LABEL)

def autoTrain():
    epoch = 60
    step_by_epoch = 430 
    (train_gen,validation_gen) = preprocessing(img_dir)
    history = trainModel(train_gen, validation_gen, epoch, step_by_epoch, "model/best.h5")
    plotTrain(history,epoch)

def loadModel(path):
    return load_model(path)

def predict(models,path,mode='category'):
    np.set_printoptions(suppress=True)
    #label list
    label = LABEL

    image = tf.keras.preprocessing.image.load_img(path)
    input_arr = keras.preprocessing.image.img_to_array(image)
    input_arr = input_arr / 255 #rescale
    input_arr=np.expand_dims(input_arr,axis=0)
    
    preds = models.predict(input_arr,verbose=True)
    pred_list = preds[0]

    i = 0
    max_prob = 0
    idx = -1

    for taux in pred_list:
        percent = round(taux*100,2)
        if(mode == "probabilities"):
            print(label[i]," : ",percent,"%")

        if(percent > max_prob):
            idx = i
            max_prob = percent
        i+=1
    
    if(mode == "category"):
        print(label[idx],"=",max_prob,"%")


def preprocessing(img_dirct):
    tf.config.experimental.enable_mlir_graph_optimization
    train_dir = os.path.join(img_dirct, 'seg_train/seg_train')
    validation_dir = os.path.join(img_dirct, 'seg_test/seg_test')

    #rescale [0-255] to [0-1] float
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        validation_dir,     
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

    return (train_generator,validation_generator)

def trainModel(train_generator,validation_generator,epochs,steps_per_epoch,path='model/first.h5'):
    
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Flatten())
    model.add(layers.Dropout(0.4))

    model.add(layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dropout(0.2))
    
    # model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    # model.add(layers.Dropout(0.2))

    # model.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    # model.add(layers.Dropout(0.2))

    model.add(layers.Dense(6, activation='softmax'))

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=50)

    model.save(path)
    return history

def plotTrain(history,epochs):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()    




def reportTestData(model):
    validation_dir = os.path.join(img_dir, 'seg_test/seg_test')
    test_datagen = ImageDataGenerator(rescale=1./255)

    validation_generator = test_datagen.flow_from_directory(
        validation_dir,     
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary',
        shuffle=False)
    Y_pred = model.predict(validation_generator)
    y_pred = np.argmax(Y_pred, axis=1)
    realpred = validation_generator.classes

    return y_pred,realpred






# def deprecated(): #first try don't pay too much attention
#     (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
#     print(train_images.shape)
#     print(train_images[0])

#     network = models.Sequential()
#     network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
#     network.add(layers.Dense(10, activation='softmax'))
#     network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

#     train_images = train_images.reshape((60000, 28 * 28))
#     print(train_images[0])
#     train_images = train_images.astype('float32') / 255*255*255
#     test_images = test_images.reshape((10000, 28 * 28))
#     test_images = test_images.astype('float32') / 255
#     train_labels = to_categorical(train_labels)
#     test_labels = to_categorical(test_labels)
#     network.fit(train_images, train_labels, epochs=5, batch_size=128)
#     test_loss, test_acc = network.evaluate(test_images, test_labels)
#     print('test_acc:', test_acc)

#     #try catch corruption
#     image = tf.keras.preprocessing.image.load_img("./image/seg_train/seg_train/buildings/0.jpg")
#     input_arr = keras.preprocessing.image.img_to_array(image)
#     print(input_arr)
#     print(len(input_arr[0]))
#     print(len(input_arr))
#     print(len(input_arr[0][0]))
#     print(input_arr.shape)
    

if __name__ == "__main__":
    main()
