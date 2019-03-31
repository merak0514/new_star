# coding=utf-8
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

import cv2


def import_data(image_path, label_path):
    """
    import data from dir and csv file
    """

    train_data = []
    with open(train_label_path) as csvfile:
        csv_reader = csv.reader(csvfile)
        data_header = next(csv_reader)

        for row in csv_reader:
            train_data.append(row)

    info = train_data[index]

    return train_data, info


def change_name(path, chr):
    """
    change filename_a.jpg to filename_b.jpg
    """
    list_p = list(path)
    list_p[-5] = chr
    return ''.join(list_p)


def create_name(path, name):
    img_name = os.path.join(path, name[0:2])
    # img_name = os.path.join(img_name, name)
    # print(img_name)
    img_list = os.listdir(img_name)
    img_list_3 = []  # img list contains 3 img a b c
    img_list_ = []
    # print(img_list)
    for img in img_list:
        if name in img:
            # print(img)
            img_list_.append(img)
            temp = os.path.join(img_name, img)
            img_list_3.append(temp)
            
    return img_list_3, img_list_


def flip_img_hori(img, x, y, num):
    img1 = img.copy()
    max_rows = img.shape[1]
    max_cols = img.shape[0]
    img1 = cv2.flip(img1, 1, dst=None) # horizontally
    x1 = max_rows - x
    y1 = y
    img1 = crop_img(img1, x1, y1, 60)
    return img1


def flip_img_verti(img, x, y, num):
    img1 = img.copy()
    max_rows = img.shape[1]
    max_cols = img.shape[0]
    img1 = cv2.flip(img1, 0, dst=None) # vertically
    x1 = x
    y1 = max_cols - y
    img1 = crop_img(img1, x1, y1, 60)
    return img1


def crop_img(img, x, y, num):
    max_rows = img.shape[1]
    max_cols = img.shape[0]
    x_min = int(max(x - num/2 , 0))
    x_max = int(min(x + num/2 , max_rows))
    y_min = int(max(y - num/2 , 0))
    y_max = int(min(y + num/2 , max_cols))
    if x_min == 0: 
        x_max = num
    elif x_max == max_rows:
        x_min = x_max - num
    
    if y_min == 0: 
        y_max = num
    elif y_max == max_cols:
        y_min = y_max - num

    print("CropShape\tX:", x_min, x, x_max, max_rows,
    '\n\t\tY:',y_min, y, y_max, max_cols)
    img1 = img.copy()
    img1 = img1[y_min : y_max ,x_min : x_max]
    return img1, x-x_min, y-y_min


def crop_and_save(path, info, img_list, img_path, nums, issave=False):
    """
    path: locations
    nums: the shape of the images
    """
    name, x, y, label = info[0], int(info[1]), int(info[2]), info[3]
 
    for i in range(3):
        i=2 # !!!
        filepath = img_path[i]
        filename = img_list[i]
        img = cv2.imread(filepath)
        img = cv2.circle(img, (x, y), 5, (0, 255, 0), 1)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # --- manipulate img ---
        img1, x1, y1 = flip_img_verti(img, x, y, nums)
        print("NewLabel:", x1, y1)
        cv2.imshow('img', img)
        cv2.imshow('cropped', img1)
        cv2.waitKey(0)

        if issave:
            save_path = path
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            cv2.imwrite(save_path+'\\'+os.path.splitext(filename)[0]+"_crop_"+os.path.splitext(filename)[1],
                        img1) # change this with manipulation!!

            print(save_path+'\\'+os.path.splitext(filename)[0]+"_crop_"+os.path.splitext(filename)[1])



if __name__ == "__main__":

    train_image_path = './af2019-cv-training-20190312/'
    train_label_path = './af2019-cv-training-20190312/list.csv'
    index = 163

    train_data, info = import_data(train_image_path, train_label_path)
    name, x, y, label = info[0], int(info[1]), int(info[2]), info[3]
    img_path, img_list = create_name(train_image_path, name) # absolute and relative location of 3 imgs
    # print(img_list)
    
    crop_and_save('./new', info, img_list, img_path, 60)

    img_a = cv2.imread(str(img_path[0]))
    img_b = cv2.imread(str(img_path[1]))
    img_c = cv2.imread(str(img_path[2]))
    img_b1 = img_b.copy()
    img_c1 = img_c.copy()
    img_b1 = cv2.circle(img_b1, (x, y), 5, (0, 255, 0), 1)
    img_c1 = cv2.circle(img_c1, (x, y), 5, (0, 255, 0), 1)


    cv2.imshow('img_a', img_a)
    cv2.imshow('img_c', img_c1)
    cv2.imshow('img_b', img_b1)

    cv2.waitKey(0)
    # print("Finished")

