import numpy as np
import pandas as pd
import random

import cv2
import os

root_dir = r'D:\paper\Self_Supervised_Learning\codes\self_supervised_learning_pytorch'
# images_dir = os.path.join(root_dir,'dataset','flowers')
images_dir = os.path.join(root_dir,'dataset','cifar10','train')

data_dir_list = os.listdir(images_dir)
print ('the data list is: ',data_dir_list)

# Assigning labels to each flower category
num_classes = 10

labels=[]

for i in range(0, num_classes):
    labels.append(i)

labels_name=dict(zip(data_dir_list,labels))
# labels_name={'daisy':0,'dandelion':1,'rose':2,'sunflower':3,'tulip':4}

# create two dataframes one for train and other for test with 3 columns as filename,label and classname
train_df = pd.DataFrame(columns=['FileName', 'Label', 'ClassName'])
val_df = pd.DataFrame(columns=['FileName', 'Label', 'ClassName'])

# test_df = pd.DataFrame(columns=['FileName', 'Label', 'ClassName'])

# number of images to take for test data from each flower category
num_images_for_test = 500

# Here data_dir_list = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
# Loop over every flower category
for dataset in data_dir_list:
    # load the list of image names in each of the flower category
    img_list = os.listdir(os.path.join(images_dir, dataset))
    print('Loading the images of dataset-' + '{}\n'.format(dataset))
    label = labels_name[dataset]
    num_img_files = len(img_list)
    num_corrupted_files = 0
    val_list_index = random.sample(range(1, num_img_files - 1), num_images_for_test)
    # test_list_index = random.sample(range(1, num_img_files - 1), num_images_for_test)

    # read each file and if it is corrupted exclude it , if not include it in either train or test data frames
    for i in range(num_img_files):
        img_name = img_list[i]
        img_filename = os.path.join(images_dir, dataset, img_name)
        try:
            input_img = cv2.imread(img_filename)
            img_shape = input_img.shape
            # if i in test_list_index:
            if i in val_list_index:
                val_df = val_df.append({'FileName': img_filename, 'Label': label, 'ClassName': dataset},
                                         ignore_index=True)
                # test_df = test_df.append({'FileName': img_filename, 'Label': label, 'ClassName': dataset},
                #                          ignore_index=True)
            else:
                train_df = train_df.append({'FileName': img_filename, 'Label': label, 'ClassName': dataset},
                                           ignore_index=True)
        except:
            print('{} is corrupted\n'.format(img_filename))
            num_corrupted_files += 1

    print('Read {0} images out of {1} images from data dir {2}\n'.format(num_img_files - num_corrupted_files,
                                                                         num_img_files, dataset))

print('completed reading all the image files and assigned labels accordingly')

dest_path=os.path.join('D:\paper\Self_Supervised_Learning\codes\self_supervised_learning_pytorch','dataset','annotations')
if not os.path.exists(dest_path):
    os.mkdir(dest_path)

train_df.to_csv(os.path.join(dest_path,'cifar10_recognition_train.csv'))
val_df.to_csv(os.path.join(dest_path,'cifar10_recognition_val.csv'))
# test_df.to_csv(os.path.join(dest_path,'flowers_recognition_test.csv'))
print('The train and val csv files are saved')

