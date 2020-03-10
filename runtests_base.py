import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import cv2
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

from sklearn.preprocessing import LabelEncoder

from keras.utils import np_utils

from keras import models
from keras.models import load_model

from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Flatten
from keras.optimizers import RMSprop

finalWidth = 100
finalHeight = 100

base_train_dir = "fv_struct\\train"
base_test_dir = "fv_struct\\test"

def prepImageDir(dirPath):
    images_rgb = []
    images_g = []
    labels = []
    for root, dirs, files in os.walk(dirPath):
        for name in files:
            dir_names = root.split("\\")
            image_rgb = cv2.imread (os.path.join(root, name))
            image_g = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
            images_rgb.append(image_rgb)
            images_g.append(image_g)
            labels.append("u_%s" % dir_names[2])
    images_rgb_np = np.array(images_rgb)
    images_g_np = np.array(images_g)
    nsamples, nx, ny = images_g_np.shape
    images_g_np = images_g_np.reshape((nsamples, nx*ny))  
    labels_np = np.array(labels) 
    return images_rgb_np, images_g_np, labels_np

X_train_vgg_base, X_train_base, Y_train_base = prepImageDir(base_train_dir)
X_test_vgg_base, X_test_base, Y_test_base = prepImageDir(base_test_dir) 

rf = RandomForestClassifier(n_estimators = 1000, random_state = 42)

rf.fit(X_train_base, Y_train_base)

y_rf_pred = rf.predict(X_test_base)

nb = GaussianNB()

nb.fit(X_train_base, Y_train_base)

y_nb_pred = nb.predict(X_test_base)

encoder = LabelEncoder()
encoder = encoder.fit(Y_train_base)

Y_train_base_encoded = encoder.transform(Y_train_base)
Y_train_base_encoded_dummy = np_utils.to_categorical(Y_train_base_encoded)

if( os.path.exists("vgg_base_model.h5") ):
    vgg_base_model = load_model("vgg_base_model.h5")
else: 
    vgg_layer = VGG16(weights="imagenet", include_top=False, input_shape=(240, 320, 3))
    
    for layer in vgg_layer.layers[:-4]:
        layer.trainable = False
        
    for layer in vgg_layer.layers:
        print(layer, layer.trainable)
        
    vgg_base_model = models.Sequential()
    
    vgg_base_model.add(vgg_layer)
    vgg_base_model.add(Flatten())
    vgg_base_model.add(Dense(106, activation='softmax'))
       
    vgg_base_model.compile(loss='categorical_crossentropy',
                    optimizer=RMSprop(lr=1e-4),
                    metrics=['accuracy'])
    
    vgg_base_model.fit(
           x=X_train_vgg_base,
           y=Y_train_base_encoded_dummy,
           epochs=30,
           verbose=1)
    
    vgg_base_model.save("vgg_base_model.h5")

vgg_base_pred = vgg_base_model.predict_classes(X_test_vgg_base)

Y_base_vgg_pred = encoder.inverse_transform(vgg_base_pred)
    
print("Base RF Accuracy:",metrics.accuracy_score(Y_test_base, y_rf_pred))
print("Base RF Precision:",metrics.precision_score(Y_test_base, y_rf_pred, average='macro'))
print("Base RF Recall:",metrics.recall_score(Y_test_base, y_rf_pred, average='macro'))
print("Base RF F1 Score:",metrics.f1_score(Y_test_base, y_rf_pred, average='macro'))

print("")

print("Base NB Accuracy:",metrics.accuracy_score(Y_test_base, y_nb_pred))
print("Base NB Precision:",metrics.precision_score(Y_test_base, y_nb_pred, average='macro'))
print("Base NB Recall:",metrics.recall_score(Y_test_base, y_nb_pred, average='macro'))
print("Base NB F1 Score:",metrics.f1_score(Y_test_base, y_nb_pred, average='macro'))

print("")

print("Base VGG Accuracy:",metrics.accuracy_score(Y_test_base, Y_base_vgg_pred))
print("Base VGG Precision:",metrics.precision_score(Y_test_base, Y_base_vgg_pred, average='macro'))
print("Base VGG Recall:",metrics.recall_score(Y_test_base, Y_base_vgg_pred, average='macro'))
print("Base VGG F1 Score:",metrics.f1_score(Y_test_base, Y_base_vgg_pred, average='macro'))



            