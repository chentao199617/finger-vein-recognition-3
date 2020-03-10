import os

import cv2

#directory paths
fold = "fvdir"
struct_dir = "fv_struct"
struct_train_dir = "fv_struct\\train"
struct_test_dir = "fv_struct\\test"

start_individual = 0 #change this variable to select last individual to sample
final_individual = 106 #change this variable to select last individual to sample

count = 0
startcount = start_individual * 6
stopcount = final_individual * 6

for root, dirs, files in os.walk(fold):
    for name in files:
        if name.endswith(".bmp") and "index" in name and "left" in root and count < stopcount and count >= startcount:
        	# load image as grayscale
            img_data_gs = cv2.imread(os.path.join(root, name), cv2.IMREAD_GRAYSCALE)
            
            # create sampled directory
            dir_names = root.split("\\")
            if(not os.path.exists(struct_dir)):
                os.makedirs(struct_dir)
                os.makedirs(struct_train_dir)
                os.makedirs(struct_test_dir)
            
            if("index_1" in name or "index_2" in name or "index_3" in name or "index_4" in name ):
                if(not os.path.exists("%s\\%s" % (struct_train_dir, dir_names[1]))):
                    os.makedirs("%s\\%s" % (struct_train_dir, dir_names[1]))
                file2save = os.path.join( struct_train_dir, dir_names[1], name)
                cv2.imwrite(file2save, img_data_gs)
                
            if("index_5" in name or "index_6" in name):
                if(not os.path.exists("%s\\%s" % (struct_test_dir, dir_names[1]))):
                    os.makedirs("%s\\%s" % (struct_test_dir, dir_names[1]))
                file2save = os.path.join(struct_test_dir, dir_names[1], name)
                cv2.imwrite(file2save, img_data_gs)
            
            count=count+1

    

            