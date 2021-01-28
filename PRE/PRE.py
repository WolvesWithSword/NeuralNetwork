import tensorflow as tf
import os
from keras.preprocessing.image import ImageDataGenerator

#permet de faire le preprocessing des image
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