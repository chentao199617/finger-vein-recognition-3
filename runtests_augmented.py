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
from keras.layers import Dense, Flatten
from keras.optimizers import RMSprop, Adam

finalWidth = 100
finalHeight = 100

proc_train_dir = "fv_proc\\train"
proc_test_dir = "fv_proc\\test"

aug_train_dir = "fv_aug\\train"
aug_test_dir = "fv_aug\\test"

def prepImageDir(dirPath):
    images_rgb = []
    images_g = []
    labels = []
    for root, dirs, files in os.walk(dirPath):
        for name in files:
            dir_names = root.split("\\")
            image_rgb = cv2.imread (os.path.join(root, name))
            image_g = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
            # add below line because features for both proc and aug images need to match
            image_rgb = cv2.resize(image_rgb, (finalWidth, finalHeight)) 
            image_g = cv2.resize(image_g, (finalWidth, finalHeight)) 
            images_rgb.append(image_rgb)
            images_g.append(image_g)
            labels.append("u_%s" % dir_names[2])
    images_rgb_np = np.array(images_rgb)
    images_g_np = np.array(images_g)
    nsamples, nx, ny = images_g_np.shape
    images_g_np = images_g_np.reshape((nsamples, nx*ny))  
    labels_np = np.array(labels) 
    return images_rgb_np, images_g_np, labels_np

X_train_vgg_aug, X_train_aug, Y_train_aug = prepImageDir(aug_train_dir)
X_test_vgg_aug, X_test_aug, Y_test_aug = prepImageDir(aug_test_dir) 
X_test_vgg_proc, X_test_proc, Y_test_proc = prepImageDir(proc_test_dir) 

rf_aug = RandomForestClassifier(n_estimators = 1000, random_state = 42)
rf_aug.fit(X_train_aug, Y_train_aug)
y_aug_rf_proc_pred = rf_aug.predict(X_test_proc)
y_aug_rf_aug_pred = rf_aug.predict(X_test_aug)

nb_aug = GaussianNB()
nb_aug.fit(X_train_aug, Y_train_aug)
y_aug_nb_proc_pred = nb_aug.predict(X_test_proc)
y_aug_nb_aug_pred = nb_aug.predict(X_test_aug)

encoder = LabelEncoder()
encoder = encoder.fit(Y_train_aug)

Y_train_aug_encoded = encoder.transform(Y_train_aug)
Y_train_aug_encoded_dummy = np_utils.to_categorical(Y_train_aug_encoded)

if( os.path.exists("vgg_aug_model.h5") ):
    vgg_aug_model = load_model("vgg_aug_model.h5")
else: 
    vgg_layer_aug = VGG16(weights="imagenet", include_top=False, input_shape=(finalWidth, finalHeight,3))
    
    for layer in vgg_layer_aug.layers[:-4]:
        layer.trainable = False
        
    vgg_aug_model = models.Sequential()
    
    vgg_aug_model.add(vgg_layer_aug)
    vgg_aug_model.add(Flatten())
    vgg_aug_model.add(Dense(106, activation='softmax'))
           
    vgg_aug_model.compile(loss='categorical_crossentropy',
                    optimizer=RMSprop(lr=1e-4),
                    metrics=['accuracy'])
    
    vgg_aug_model.fit(
           x=X_train_vgg_aug,
           y=Y_train_aug_encoded_dummy,
           epochs=30,
           verbose=1)
    
    vgg_aug_model.save("vgg_aug_model.h5")
    
if( os.path.exists("vgg_aug_model_tune.h5") ):
    vgg_aug_model_tune = load_model("vgg_aug_model_tune.h5")
else: 
    vgg_layer_aug_tune = VGG16(weights="imagenet", include_top=False, input_shape=(finalWidth, finalHeight,3))
    
    for layer in vgg_layer_aug_tune.layers[:-3]:
        layer.trainable = False
        
    vgg_aug_model_tune = models.Sequential()
    
    vgg_aug_model_tune.add(vgg_layer_aug_tune)
    vgg_aug_model_tune.add(Flatten())
    vgg_aug_model_tune.add(Dense(106, activation='softmax'))
    
    vgg_aug_model_tune.compile(loss='categorical_crossentropy',
                    optimizer=Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, amsgrad=True),
                    metrics=['accuracy'])
    
    vgg_aug_model_tune.fit(
           x=X_train_vgg_aug,
           y=Y_train_aug_encoded_dummy,
           epochs=30,
           verbose=1)
    
    vgg_aug_model_tune.save("vgg_aug_model_tune.h5")

aug_vgg_proc_pred = vgg_aug_model.predict_classes(X_test_vgg_proc)
aug_tune_vgg_proc_pred = vgg_aug_model_tune.predict_classes(X_test_vgg_proc)
aug_vgg_aug_pred = vgg_aug_model.predict_classes(X_test_vgg_aug)
aug_tune_vgg_aug_pred = vgg_aug_model_tune.predict_classes(X_test_vgg_aug)

Y_aug_vgg_proc_pred = encoder.inverse_transform(aug_vgg_proc_pred)
Y_aug_tune_vgg_proc_pred = encoder.inverse_transform(aug_tune_vgg_proc_pred)
Y_aug_vgg_aug_pred = encoder.inverse_transform(aug_vgg_aug_pred)
Y_aug_tune_vgg_aug_pred = encoder.inverse_transform(aug_tune_vgg_aug_pred)

print("Aug RF Proc Accuracy:",metrics.accuracy_score(Y_test_proc, y_aug_rf_proc_pred))
print("Aug RF Proc Precision:",metrics.precision_score(Y_test_proc, y_aug_rf_proc_pred, average='macro'))
print("Aug RF Proc Recall:",metrics.recall_score(Y_test_proc, y_aug_rf_proc_pred, average='macro'))
print("Aug RF Proc F1 Score:",metrics.f1_score(Y_test_proc, y_aug_rf_proc_pred, average='macro'))

print("")

print("Aug RF Aug Accuracy:",metrics.accuracy_score(Y_test_aug, y_aug_rf_aug_pred))
print("Aug RF Aug Precision:",metrics.precision_score(Y_test_aug, y_aug_rf_aug_pred, average='macro'))
print("Aug RF Aug Recall:",metrics.recall_score(Y_test_aug, y_aug_rf_aug_pred, average='macro'))
print("Aug RF Aug F1 Score:",metrics.f1_score(Y_test_aug, y_aug_rf_aug_pred, average='macro'))

print("")

print("Aug NB Proc Accuracy:",metrics.accuracy_score(Y_test_proc, y_aug_nb_proc_pred))
print("Aug NB Proc Precision:",metrics.precision_score(Y_test_proc, y_aug_nb_proc_pred, average='macro'))
print("Aug NB Proc Recall:",metrics.recall_score(Y_test_proc, y_aug_nb_proc_pred, average='macro'))
print("Aug NB Proc F1 Score:",metrics.f1_score(Y_test_proc, y_aug_nb_proc_pred, average='macro'))

print("")

print("Aug NB Aug Accuracy:",metrics.accuracy_score(Y_test_aug, y_aug_nb_aug_pred))
print("Aug NB Aug Precision:",metrics.precision_score(Y_test_aug, y_aug_nb_aug_pred, average='macro'))
print("Aug NB Aug Recall:",metrics.recall_score(Y_test_aug, y_aug_nb_aug_pred, average='macro'))
print("Aug NB Aug F1 Score:",metrics.f1_score(Y_test_aug, y_aug_nb_aug_pred, average='macro'))

print("")

print("Aug VGG Proc Accuracy:",metrics.accuracy_score(Y_test_proc, Y_aug_vgg_proc_pred))
print("Aug VGG Proc Precision:",metrics.precision_score(Y_test_proc, Y_aug_vgg_proc_pred, average='macro'))
print("Aug VGG Proc Recall:",metrics.recall_score(Y_test_proc, Y_aug_vgg_proc_pred, average='macro'))
print("Aug VGG Proc F1 Score:",metrics.f1_score(Y_test_proc, Y_aug_vgg_proc_pred, average='macro'))

print("")

print("Aug VGG Aug Accuracy:",metrics.accuracy_score(Y_test_aug, Y_aug_vgg_aug_pred))
print("Aug VGG Aug Precision:",metrics.precision_score(Y_test_aug, Y_aug_vgg_aug_pred, average='macro'))
print("Aug VGG Aug Recall:",metrics.recall_score(Y_test_aug, Y_aug_vgg_aug_pred, average='macro'))
print("Aug VGG Aug F1 Score:",metrics.f1_score(Y_test_aug, Y_aug_vgg_aug_pred, average='macro'))

print("")

print("Aug Tune VGG Proc Accuracy:",metrics.accuracy_score(Y_test_proc, Y_aug_tune_vgg_proc_pred))
print("Aug Tune VGG Proc Precision:",metrics.precision_score(Y_test_proc, Y_aug_tune_vgg_proc_pred, average='macro'))
print("Aug Tune VGG Proc Recall:",metrics.recall_score(Y_test_proc, Y_aug_tune_vgg_proc_pred, average='macro'))
print("Aug Tune VGG Proc F1 Score:",metrics.f1_score(Y_test_proc, Y_aug_tune_vgg_proc_pred, average='macro'))

print("")

print("Aug Tune VGG Aug Accuracy:",metrics.accuracy_score(Y_test_aug, Y_aug_tune_vgg_aug_pred))
print("Aug Tune VGG Aug Precision:",metrics.precision_score(Y_test_aug, Y_aug_tune_vgg_aug_pred, average='macro'))
print("Aug Tune VGG Aug Recall:",metrics.recall_score(Y_test_aug, Y_aug_tune_vgg_aug_pred, average='macro'))
print("Aug Tune VGG Aug F1 Score:",metrics.f1_score(Y_test_aug, Y_aug_tune_vgg_aug_pred, average='macro'))



            