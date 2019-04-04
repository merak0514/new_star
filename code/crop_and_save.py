# coding=utf-8
import csv
import os
import random

import matplotlib.pyplot as plt
import numpy as np

import cv2


def import_data(index, label_path):
    """
    从csv文件中导入数据
    """

    train_data = []
    with open(label_path) as csvfile:
        csv_reader = csv.reader(csvfile)
        data_header = next(csv_reader)

        for row in csv_reader:
            train_data.append(row)

    info = train_data[index]

    # 创建写入的文件
    if not os.path.exists('list.csv'):
        with open('new_labels.csv', 'w', newline='') as csvfile:  # 为write_data 做准备
            csv_writer = csv.writer(csvfile)
            csv_writer = csv_writer.writerow(
                ["index", "filename", "x", "y", "label"])

    return train_data, info


def write_data(path, index, filename, text, x, y, label):
    """
    写入csv文件
    """
    string = os.path.splitext(filename)[0]+text
    write_list = [index, string, x, y, label]
    with open('../cut_data/new_labels.csv', 'a+', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer = csv_writer.writerow(write_list)


def change_name(path, chr):
    """
   改文件名：filename_a.jpg to filename_b.jpg
    """
    list_p = list(path)
    list_p[-5] = chr
    return ''.join(list_p)


def create_name(path, name):
    """
    提取3个文件的相对地址和文件名，放到列表里面
    :return: img_list_3: 返回文件的相对路径
    :return: img_list: 返回文件名
    """
    img_name = os.path.join(path, name[0:2])  # 定位文件夹（用两个字符）
    # img_name = os.path.join(img_name, name)
    # print(img_name)
    img_list = os.listdir(img_name)  # 定位文件夹里面的图片
    img_list_3 = []  # img list contains 3 img a b c
    img_list_ = []
    # print(img_list)
    for img in img_list:
        if name in img:
            # print(img)
            img_list_.append(img)  # 3张图片的名字
            temp = os.path.join(img_name, img)
            img_list_3.append(temp)  # 3张图片的相对地址

    return img_list_3, img_list_


def save_img(path, filename, img, text):
    """
    保存图片，文件名命名格式：地址 + 文件名 + 修改方式 + 后缀
    """
    save_path = path+'\\' + \
        os.path.splitext(filename)[0]+text+os.path.splitext(filename)[1]
    if not os.path.exists(path):
        os.makedirs(path)
    cv2.imwrite(save_path, img)  # change this with manipulation!!

    print(save_path)


def flip_img_hori(img, x, y, num, seed=0, israndom=False):
    """
    水平翻转
    """
    img1 = img.copy()
    max_rows = img.shape[1]
    # max_cols = img.shape[0]
    img1 = cv2.flip(img1, 1, dst=None)  # horizontally
    x1 = max_rows - x
    y1 = y
    img1, x2, y2 = crop_img(img1, x1, y1, num, seed, israndom)  # 剪
    return img1, x2, y2 
 

def flip_img_verti(img, x, y, num, seed=0, israndom=False):
    """
    竖直翻转
    """
    img1 = img.copy()
    # max_rows = img.shape[1]
    max_cols = img.shape[0]
    img1 = cv2.flip(img1, 0, dst=None)  # vertically
    x1 = x
    y1 = max_cols - y
    img1 = crop_img(img1, x1, y1, num, seed, israndom)  # 剪
    return img1, x1, y1


def rotate_img(img, x, y, degree, num, seed=0, israndom=False):
    """
    任意角度旋转图片，num是需要旋转后并剪出的图片的大小
    """
    img1 = img.copy()
    max_rows = img.shape[1]
    max_cols = img.shape[0]
    Matrix = cv2.getRotationMatrix2D((x, y), degree, 1)  # rotate matrix
    img1 = cv2.warpAffine(img, Matrix, (max_rows, max_cols))

    # cv2.imshow('img1', img1)
    img2, x1, y1 = crop_img(img1, x, y, num, seed, israndom)  # 剪
    return img2, x1, y1


def crop_img(img, x, y, num, seed=0, israndom=False):
    """
    剪图片，num是大小，这里先默认只放中间，后面可改动加#部位可以任意位置
    剪出来的图片中星体在(a,b)位置，为了保证数据多样性加入随机数处理、
    """
    if israndom:
        random.seed(seed)
        a = int(random.uniform(0, 1) * num)  # 均匀分布，x坐标
        b = int(random.uniform(0, 1) * num)  # 均匀分布，y坐标
    else:
        a = int(0.5 * num)
        b = int(0.5 * num)

    max_rows = img.shape[1]
    max_cols = img.shape[0]

    # --- crop ---
    x_min = int(max(x - a, 0))
    x_max = int(min(x - a + num, max_rows))
    y_min = int(max(y - b, 0))
    y_max = int(min(y - b + num, max_cols))

    if x_min == 0:
        x_max = num
    elif x_max == max_rows:
        x_min = x_max - num

    if y_min == 0:
        y_max = num
    elif y_max == max_cols:
        y_min = y_max - num
    # --- crop ---

    print("CropShape\tX:", x_min, x, x_max, max_rows,
          '\n\t\tY:', y_min, y, y_max, max_cols)
    img1 = img.copy()
    img1 = img1[y_min: y_max, x_min: x_max]
    return img1, x-x_min, y-y_min


def crop_and_save(path, info, img_list, img_path, nums):
    """
    剪辑并保存的函数
    更新：可以实现目标点在新图片中的随机位置，采用随机数形式实现
    需要使abc三张图片裁剪方式相同，采取改变种子的方式实现
    只需要修改后面crop_img参数中的随机数种子即可

    :param path: 图片的路径
    :param info: csv中的每一行 (name, x, y, label)
    :param img_list: 3个文件名
    :param origin_pos: 3个文件的路径
    :param nums: 图片的大小（这个变量应该改成size的）
    """
    x, y, label = int(info[1]), int(info[2]), info[3]
    seed = random.randint(0, 100)  
    # 随机数种子。需要使abc三张图片裁剪方式相同，采取改变种子的方式实现

    for i in range(3):  # i=0 时为图片a

        filepath = img_path[i]
        filename = img_list[i]
        print("\n>>>>\t\t", i, "\t\t<<<<")
        img = cv2.imread(filepath)
        img = cv2.circle(img, (x, y), 5, (0, 255, 0), 1)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # --- 操作图片 ---
        text = "_crop"
        print("\n\n>>>>\t", text)
        img0, x1, y1 = crop_img(img, x, y, nums, seed, israndom=True)
        print("NewLabel:", x1, y1)
        write_data(path, index, filename, text, x1, y1, label)
        save_img(path, filename, img0, text)  # 文件名命名格式：地址 + 文件名 + 修改方式 + 后缀

        # text = "_rotate_60"
        # print("\n\n>>>>\t", text)
        # img1, x1, y1 = rotate_img(img, x, y, 60, nums, seed+4, israndom=True)
        # print("NewLabel:", x1, y1)
        # write_data(path, index, filename, text, x1, y1, label)
        # save_img(path, filename, img1, text)

        # text = "_rotate_45"
        # print("\n\n>>>>\t", text)
        # img1, x1, y1 = rotate_img(img, x, y, 45, nums, seed+2, israndom=True)
        # print("NewLabel:", x1, y1)
        # write_data(path, index, filename, text, x1, y1, label)
        # save_img(path, filename, img1, text)

        # text = "_flip_hori"
        # print("\n\n>>>>\t", text)
        # img2, x1, y1 = flip_img_hori(img, x, y, 60, seed+5, israndom=True)
        # print("NewLabel:", x1, y1)
        # write_data(path, index, filename, text, x1, y1, label)
        # save_img(path, filename, img2, text)

        # text = "_flip_verti"
        # print("\n\n>>>>\t", text)
        # img3, x1, y1 = flip_img_verti(img, x, y, 60, seed+9, israndom=True)
        # print("NewLabel:", x1, y1)
        # write_data(path, index, filename, text, x1, y1, label)
        # save_img(path, filename, img3, text)

    # 下面应该写一段把label保存为csv格式的程序 (已经完成了，见write_data)


if __name__ == "__main__":

    train_image_path = '../../af2019-cv-training-20190312/'
    train_label_path = '../../af2019-cv-training-20190312/list.csv'
    for index in range(50):

        train_data, info = import_data(index, train_label_path)
        name, x, y, label = info[0], int(info[1]), int(info[2]), info[3]

        img_path, img_list = create_name(train_image_path, name)  # 图片的地址和图片的名字
        # print(img_list)

        crop_and_save('../cut_data', info, img_list, img_path, nums=100)

    # img_a = cv2.imread(str(img_path[0]))
    # img_b = cv2.imread(str(img_path[1]))
    # img_c = cv2.imread(str(img_path[2]))
    # img_b1 = img_b.copy()
    # img_c1 = img_c.copy()
    # img_b1 = cv2.circle(img_b1, (x, y), 5, (0, 255, 0), 1)
    # img_c1 = cv2.circle(img_c1, (x, y), 5, (0, 255, 0), 1)

    # cv2.imshow('img_a', img_a)
    # cv2.imshow('img_c', img_c1)
    # cv2.imshow('img_b', img_b1)

    # cv2.imshow('img', img)
    # cv2.imshow('cropped', img1)
    # cv2.waitKey(0)

    # cv2.waitKey(0)
    # print("Finished")
