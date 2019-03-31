# !/user/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 16:33
# @Author   : Merak
# @File     : test.py
# @Software : PyCharm
from pre_processing import *
train_image_path = '../af2019-cv-training-20190312/'
train_label_path = '../af2019-cv-training-20190312/list.csv'
a = '_a'
b = '_b'
c = '_c'
end = '.jpg'


def get_pos_based_on_name(name):
    train_data = []
    with open(train_label_path) as csvfile:
        csv_reader = csv.reader(csvfile)
        data_header = next(csv_reader)
        for row in csv_reader:
            train_data.append(row)
    for datum in train_data:
        if datum[0] == name:
            return int(datum[1]), int(datum[2]), datum[3]


if __name__ == '__main__':
    img_name = '00b4b8a8152a05872297ec33fecac289'

    imb = cv2.imread(''.join((train_image_path, img_name[:2], '/', img_name, b, end)))
    imb = imb[:, :, 0]
    imc = cv2.imread(''.join((train_image_path, img_name[:2], '/', img_name, c, end)))
    imc = imc[:, :, 0]

    imb = cut_too_small(imb, lambda_=1.75)
    imb = cut_too_large(imb)
    # cv2.imshow('imb', imb)
    imb = middle_filter(imb).copy()
    # cv2.imshow('imb2', imb)

    imc = cut_too_small(imc, lambda_=1.5)
    imc = cut_too_large(imc)
    imc = middle_filter(imc).copy()

    ima_cut = imb - imc
    ima_cut *= (ima_cut > 0)  # 去掉小于0
    # cv2.imshow('ima_cut', ima_cut)
    x, y, label = get_pos_based_on_name(img_name)
    print('label', label)
    img = cv2.circle(ima_cut, (x, y), 5, 255, 1)
    print(np.shape(imb))
    img_b = cv2.circle(imb, (x, y), 5, 255, 1)
    img_c = cv2.circle(imc, (x, y), 5, 255, 1)
    cv2.imshow('img', img)
    cv2.imshow('img_b', img_b)
    cv2.imshow('img_c', img_c)
    cv2.waitKey(0)


