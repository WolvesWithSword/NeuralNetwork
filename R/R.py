from tensorflow import keras
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
from PRE.PRE import *
from FIG.FIG import *
img_dir = "./image/"
LABEL = ["buildings","forest","glacier","mountain","sea","street"]

#Permet de train et d'afficher les courbe resultant de l'entrainement
def autoTrain(epoch = 60, step_by_epoch = 430,modelPath = "model.h5"):

    (train_gen,validation_gen) = preprocessing(img_dir)
    history = trainModel(train_gen, validation_gen, epoch, step_by_epoch, modelPath)
    plotTrain(history,epoch)

#permet de charger le modele
def loadModel(path):
    return load_model(path)

#permet de predire une image
def predict(models,path,mode='category'):
    np.set_printoptions(suppress=True)
    #label list
    label = LABEL

    image = tf.keras.preprocessing.image.load_img(path)
    input_arr = keras.preprocessing.image.img_to_array(image)
    input_arr = input_arr / 255 #rescale
    input_arr=np.expand_dims(input_arr,axis=0)
    print("a")
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



#permet d'entrainer le modele
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


# permet de créer les liste des prediction du jeu de test
# ainsi que les réel 
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




