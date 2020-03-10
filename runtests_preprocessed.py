import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import cv2
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

from keras import models
from keras.models import load_model
from keras.utils import np_utils
from keras.applications.vgg16 import VGG16
from keras.layers import Dense,Flatten
from keras.optimizers import RMSprop

finalWidth = 200
finalHeight = 200

proc_train_dir = "fv_proc\\train"
proc_test_dir = "fv_proc\\test"

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
  

X_train_vgg_proc, X_train_proc, Y_train_proc = prepImageDir(proc_train_dir)
X_test_vgg_proc, X_test_proc, Y_test_proc = prepImageDir(proc_test_dir) 

rf_proc = RandomForestClassifier(n_estimators = 1000, random_state = 42)
rf_proc.fit(X_train_proc, Y_train_proc)
y_rf_proc_pred = rf_proc.predict(X_test_proc)

nb_proc = GaussianNB()
nb_proc.fit(X_train_proc, Y_train_proc)
y_nb_proc_pred = nb_proc.predict(X_test_proc)

encoder = LabelEncoder()
encoder = encoder.fit(Y_train_proc)

Y_train_proc_encoded = encoder.transform(Y_train_proc)
Y_train_proc_encoded_dummy = np_utils.to_categorical(Y_train_proc_encoded)

if( os.path.exists("vgg_proc_model.h5") ):
    vgg_proc_model = load_model("vgg_proc_model.h5")
else: 
    vgg_layer_proc = VGG16(weights="imagenet", include_top=False, input_shape=(finalWidth, finalHeight,3))
    
    for layer in vgg_layer_proc.layers[:-4]:
        layer.trainable = False
        
    vgg_proc_model = models.Sequential()
    
    vgg_proc_model.add(vgg_layer_proc)
    vgg_proc_model.add(Flatten())
    vgg_proc_model.add(Dense(106, activation='softmax'))
       
    vgg_proc_model.compile(loss='categorical_crossentropy',
                    optimizer=RMSprop(lr=1e-4),
                    metrics=['accuracy'])
    
    vgg_proc_model.fit(
           x=X_train_vgg_proc,
           y=Y_train_proc_encoded_dummy,
           epochs=30,
           verbose=1)
    
    vgg_proc_model.save("vgg_proc_model.h5")

vgg_proc_pred = vgg_proc_model.predict_classes(X_test_vgg_proc)

Y_proc_vgg_pred = encoder.inverse_transform(vgg_proc_pred)

print("Proc RF Accuracy:",metrics.accuracy_score(Y_test_proc, y_rf_proc_pred))
print("Proc RF Precision:",metrics.precision_score(Y_test_proc, y_rf_proc_pred, average='macro'))
print("Proc RF Recall:",metrics.recall_score(Y_test_proc, y_rf_proc_pred, average='macro'))
print("Proc RF F1 Score:",metrics.f1_score(Y_test_proc, y_rf_proc_pred, average='macro'))

print("")

print("Proc NB Accuracy:",metrics.accuracy_score(Y_test_proc, y_nb_proc_pred))
print("Proc NB Precision:",metrics.precision_score(Y_test_proc, y_nb_proc_pred, average='macro'))
print("Proc NB Recall:",metrics.recall_score(Y_test_proc, y_nb_proc_pred, average='macro'))
print("Proc NB F1 Score:",metrics.f1_score(Y_test_proc, y_nb_proc_pred, average='macro'))

print("")

print("Proc VGG Accuracy:",metrics.accuracy_score(Y_test_proc, Y_proc_vgg_pred))
print("Proc VGG Precision:",metrics.precision_score(Y_test_proc, Y_proc_vgg_pred, average='macro'))
print("Proc VGG Recall:",metrics.recall_score(Y_test_proc, Y_proc_vgg_pred, average='macro'))
print("Proc VGG F1 Score:",metrics.f1_score(Y_test_proc, Y_proc_vgg_pred, average='macro'))



            