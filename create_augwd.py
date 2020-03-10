#import libraries we will be using
import os
import shutil
import cv2
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator

#dimensions we want out images resized to
finalWidth = 100
finalHeight = 100

#directory paths
proc_dir = "fv_proc"
aug_dir = "fv_aug"
aug_train_dir = "fv_aug\\train"
aug_test_dir = "fv_aug\\test"

#remove existing aug directory, so we make a fresh one when we run this
if(os.path.exists(aug_dir)):
    shutil.rmtree(aug_dir)
    
os.makedirs(aug_dir)
os.makedirs(aug_train_dir)
os.makedirs(aug_test_dir)

for root, dirs, files in os.walk(proc_dir):
    for name in files:
        
        img_path = os.path.join(root, name)
        dir_names = root.split("\\")
        filename_parts = name.split(".")
        fileprefix = "aug_%s" % filename_parts[0]
        
        if ("train" in root):
            if(not os.path.exists(os.path.join(aug_train_dir,dir_names[2]))):
                os.makedirs(os.path.join(aug_train_dir,dir_names[2]))
            
            #create ImageDataGenerator for augmented training images
            datagen = ImageDataGenerator(
                width_shift_range=0.10,
                height_shift_range=0.10,
                rotation_range=15,
                brightness_range=[0.5,1.1],
                zoom_range=[0.75,1.15])
            
            savedir = os.path.join(aug_train_dir,dir_names[2]) 
            tcount=10
        elif ("test" in root):
            if(not os.path.exists(os.path.join(aug_test_dir,dir_names[2]))):
                os.makedirs(os.path.join(aug_test_dir,dir_names[2]))
            
            #create ImageDataGenerator for augmented test images
            datagen = ImageDataGenerator(
                width_shift_range=0.05,
                height_shift_range=0.05,
                #horizontal_flip=True,
                rotation_range=7.5,
                brightness_range=[0.9,1.1],
                zoom_range=[0.95,1.05])
            
            savedir = os.path.join(aug_test_dir,dir_names[2])
            tcount = 3
        else:
            break
        
        img_load = load_img(img_path)  
        img_data = img_to_array(img_load)
        img_resize = cv2.resize(img_data, (finalWidth, finalHeight))
        img_sample = expand_dims(img_data,0)
        
        seed = 1
        total = 0
        for batch in datagen.flow(
                img_sample,
                save_to_dir= savedir,
                save_prefix=fileprefix,
                save_format='bmp',
                batch_size = 1,
                seed = seed):
            
            total += 1
            if total == tcount:
                break

            