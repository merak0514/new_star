# !/user/bin/env python
# -*- coding: utf-8 -*-
import csv
import os
import cv2
import math


def save_img(path, filename, img, text, x1, y1):
    """

    :param path: 文件路径
    :param filename: 文件名
    :param img:
    :param text:
    :param x1:
    :param y1:
    :return:
    """

    if not os.path.exists(path + 'label' + '/'):
        os.makedirs(path + 'label' + '/')
    txt_path = path + 'label' + '/' + filename[:-6] + text + '.txt'

    if '_b' in filename:
        path = path + 'image_b'
    elif '_c' in filename:
        path = path + 'image_c'  
    elif '_a' in filename:
        path = path + 'image_a'

    if not os.path.exists(path):
        os.makedirs(path)
    filename_ = filename[:-6]
    save_path = path + '/' + filename_ + text + '.png'
    cv2.imwrite(save_path, img)  # change this with manipulation!!
    
    print(txt_path)
    with open(txt_path, 'w') as f:
        text_w = filename_ + text + ' ' + str(x1) + ' ' + str(y1) + ' ' + '1' + '\n'
        f.write(text_w)

    print('Saved:\t' + filename + text)

    print(save_path)


def flip_img_hori(img, x, y):
    """
    水平翻转
    """
    img1 = img.copy()
    max_rows = img.shape[1]
    # max_cols = img.shape[0]
    img1 = cv2.flip(img1, 1, dst=None)  # horizontally
    x1 = max_rows - x
    y1 = y
    return img1, x1, y1


def flip_img_verti(img, x, y):
    """
    竖直翻转
    """
    img1 = img.copy()
    # max_rows = img.shape[1]
    max_cols = img.shape[0]
    img1 = cv2.flip(img1, 0, dst=None)  # vertically
    x1 = x
    y1 = max_cols - y
    return img1, x1, y1


def rotate_img(img, x, y, angle):
    """
    任意角度旋转图片，num是需要旋转后并剪出的图片的大小
    """
    img1 = img.copy()
    max_rows = img.shape[0]
    max_cols = img.shape[1]
    center = (max_rows/2, max_cols/2)
    Matrix = cv2.getRotationMatrix2D(center, angle, 1)  # rotate matrix
    img1 = cv2.warpAffine(img, Matrix, (max_rows, max_cols))

    x1 = (x - max_rows/2) * math.cos(angle * math.pi / 180) + \
        (y - max_cols/2) * math.sin(angle * math.pi / 180) + max_rows/2
    y1 = -1 * (x - max_rows/2) * math.sin(angle * math.pi / 180) + \
        (y - max_cols/2) * math.cos(angle * math.pi / 180) + max_cols/2
    return img1, int(x1), int(y1)


def import_data(label_path):
    """
    从csv文件中导入数据
    """
    train_data = []

    with open(label_path) as csvfile:
        csv_reader = csv.reader(csvfile)
        # data_header = next(csv_reader)

        for row in csv_reader:
            if row[3] == '1':
                train_data.append(row)

    # # 创建写入的文件
    # if not os.path.exists('../good_data/good_list.csv'):
    #     with open('../good_data/good_list.csv', 'w', newline='') as csvfile:  # 为write_data 做准备
    #         csv_writer = csv.writer(csvfile)
    #         csv_writer = csv_writer.writerow(
    #             ["filename", "x", "y", "label"])

    return train_data


if __name__ == "__main__":

    train_image_path = '../cut_data2/'
    train_label_path = '../cut_data2/new_labels.csv'
    save_path = '../good_data50/'

    train_data = import_data(train_label_path)
    print("导出数据:\t",train_data)

    for row in train_data:
        img_name, x, y, label = row[0], int(row[1]), int(row[2]), row[3]

        img_list = os.listdir(''.join([train_image_path, img_name[:2],'/']))
        for img_full_name in img_list:
            if img_name in img_full_name:
                img_path = ''.join([train_image_path, img_name[:2],'/',img_full_name])
    
                img = cv2.imread(img_path)

                # img = cv2.circle(img, (x, y), 5, (0, 255, 0), 1)
                # cv2.imshow('img2', img)
                # cv2.waitKey(0)
                text = '_origin'
                save_img(save_path, img_full_name, img, text, x, y)

                text = '_flip_hori'
                img1, x1, y1 = flip_img_hori(img, x, y)
                save_img(save_path, img_full_name, img1, text, x1, y1)

                text = '_flip_verti'
                img2, x1, y1 = flip_img_verti(img, x, y)
                save_img(save_path, img_full_name, img2, text, x1, y1)

                text = '_flip_rotate_90'
                img2, x1, y1 = rotate_img(img2, x1, y1, 90)
                save_img(save_path, img_full_name, img2, text, x1, y1)

                text = '_flip_rotate_270'
                img2, x1, y1 = rotate_img(img2, x1, y1, 180)
                save_img(save_path, img_full_name, img2, text, x1, y1)

                text = '_rotate_90'
                img3, x1, y1 = rotate_img(img, x, y, 90)
                save_img(save_path, img_full_name, img3, text, x1, y1)

                # img3 = cv2.circle(img3, (x1, y1), 5, (0, 255, 0), 1)
                # cv2.imshow('img3', img3)
                # cv2.waitKey(0)

                text = '_rotate_180'
                img3, x1, y1 = rotate_img(img3, x1, y1, 90)
                save_img(save_path, img_full_name, img3, text, x1, y1)

                text = '_rotate_270'
                img3, x1, y1 = rotate_img(img3, x1, y1, 180)
                save_img(save_path, img_full_name, img3, text, x1, y1)

        # img3 = cv2.circle(img3, (x1, y1), 5, (0, 255, 0), 1)
        # cv2.imshow('img3', img3)
        # cv2.waitKey(
    print("All Done !!")
