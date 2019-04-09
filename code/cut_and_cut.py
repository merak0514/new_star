# !/user/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 19:22
# @Author   : Merak
# @File     : cut_and_cut.py
# @Software : PyCharm
import cv2
import numpy as np
import os
import csv
IMAGE_B = 'image_b/'
IMAGE_C = 'image_c/'
b = '_b'
c = '_c'
end2 = '.png'


def cut(image_origin_1, image_origin_2, origin_pos, size=(25, 25)):

    x_choices = np.arange(4) * 25
    y_choices = np.arange(4) * 25

    combines = []  # 所有开始切的坐标的合集
    labels = []  # 和combines一一对应，为每一个坐标所对应图片中
    for i in x_choices:
        for j in y_choices:
            combines.append((i, j))  # 相当于添加每个方框的开始坐标
            if (i <= origin_pos[0] < i+size[0]) and (j <= origin_pos[1] < j+size[1]):
                print('yyyyy')
                input(1)
                labels.append((origin_pos[0]-i, origin_pos[1]-j))
            else:
                labels.append(0)
    # print(combines)
    # print(labels)

    images_1 = []
    for position in combines:
        image = image_origin_1[position[0]: position[0] + size[0], position[1]: position[1] + size[1]]
        images_1.append(image)
    images_2 = []
    for position in combines:
        image = image_origin_2[position[0]: position[0] + size[0], position[1]: position[1] + size[1]]
        images_2.append(image)

    return images_1, images_2, labels


def cut_all(csv_file,
            good_train_set_label_path='../good_data/label/',
            good_train_set_path='../good_data/',
            new_good_train_set_path='../good_data2/',
            ):
    good_data = []
    all_good_txt = os.listdir(good_train_set_label_path)
    for txt in all_good_txt:
        txt_data = open(good_train_set_label_path+txt).readline().split(' ')
        good_data.append((txt_data[0], (int(txt_data[1]), int(txt_data[2]))))
    for i in good_data:
        print(i)
        image_name = i[0]
        pos = i[1]
        img_b = cv2.imread(''.join((good_train_set_path, IMAGE_B, image_name, end2)))
        img_b = np.array(img_b[:, :, 0], np.uint8)
        img_c = cv2.imread(''.join((good_train_set_path, IMAGE_C, image_name, end2)))
        img_c = np.array(img_c[:, :, 0], np.uint8)
        cut_images_b, cut_images_c, labels = cut(img_b, img_c, pos)

        path = ''.join([new_good_train_set_path, image_name[:2]])
        if not os.path.exists(path):
            print(path)
            os.makedirs(path)

        for i in range(len(cut_images_b)):
            image = cut_images_b[i]
            im_path = ''.join([path, '/', image_name, '_', str(i), b, end2])
            cv2.imwrite(im_path, image)
            data_row = [image_name + '_' + str(i), labels[i]]
            csv.writer(csv_file).writerow(data_row)

        for i in range(len(cut_images_c)):
            image = cut_images_c[i]
            im_path = ''.join([path, '/', image_name, '_', str(i), c, end2])
            cv2.imwrite(im_path, image)


if __name__ == '__main__':
    csv_file = open('../good_data2/labels.csv', 'a+')
    cut_all(csv_file)
    csv_file.close()
