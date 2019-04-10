# !/user/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 8:07
# @Author   : Merak
# @File     : evaluation.py
# @Software : PyCharm
import os
import resnet
import re
import torch
import classification
import numpy as np
import cv2
os.environ['CUDA_VISIBLE_DEVICES'] = '8'

resnet18 = resnet.resnet18().cuda()
resnet18.eval()

model_path = './model2/'
bad_train_data_set_path = '../../cut_data/'
good_train_data_set_path = '../../good_data/'
IMAGE_B = 'image_b/'
IMAGE_C = 'image_c/'
b = '_b'
c = '_c'
end = '.jpg'
end2 = '.png'
TEST_SIZE = 1000


if __name__ == '__main__':
    print('start evaluating!')
    model_name = classification.find_newest_model()
    model = torch.load(model_path+model_name)
    resnet18.load_state_dict(model['model_state_dict'])
    print('train set accuracy: ' + str(model['accuracy']))
    _, bad_test_data = classification.import_bad_data()
    _, good_test_data = classification.import_good_data()
    np.random.shuffle(bad_test_data)
    np.random.shuffle(good_test_data)

    print('testing good data')
    good_labels = torch.ones(TEST_SIZE).cuda().type(torch.LongTensor)
    good_images = torch.Tensor([]).cuda()
    for i in range(TEST_SIZE):
        image_name = good_test_data[i]
        image_b = torch.Tensor(
            cv2.imread(''.join([good_train_data_set_path, IMAGE_B, image_name, end2])))[:, :, 0]
        image_c = torch.Tensor(
            cv2.imread(''.join([good_train_data_set_path, IMAGE_C, image_name, end2])))[:, :, 0]
        image_combine = torch.cat((image_b.unsqueeze(2), image_c.unsqueeze(2)), dim=2).cuda()  # 作为二通道的输入
        good_images = torch.cat((good_images, image_combine.unsqueeze(0)), 0)
    good_images = good_images.permute((0, 3, 1, 2))
    good_outputs = resnet18.forward(good_images)
    predict = (good_outputs[:, 1] > good_outputs[:, 0]).type(torch.LongTensor)
    accuracy = sum((predict == good_labels).type(torch.FloatTensor)) / len(predict)
    print(''.join(['Have tested ', str(TEST_SIZE), ' good pictures, accuracy = ', str(accuracy)]))
        
    print('testing bad data')
    bad_labels = torch.zeros(TEST_SIZE).type(torch.LongTensor)
    bad_images = torch.Tensor([]).cuda()
    for i in range(TEST_SIZE):
        image_name = bad_test_data[i]
        image_b = torch.Tensor(
            cv2.imread(''.join([bad_train_data_set_path, image_name[:2], '/', image_name, b, end])))[:, :, 0]
        image_c = torch.Tensor(
            cv2.imread(''.join([bad_train_data_set_path, image_name[:2], '/', image_name, c, end])))[:, :, 0]
        image_combine = torch.cat((image_b.unsqueeze(2), image_c.unsqueeze(2)), dim=2).cuda()  # 作为二通道的输入
        bad_images = torch.cat((bad_images, image_combine.unsqueeze(0)), 0)
    bad_images = bad_images.permute((0, 3, 1, 2))
    bad_outputs = resnet18.forward(bad_images)
    predict = (bad_outputs[:, 1] > bad_outputs[:, 0]).cuda().type(torch.LongTensor)
    accuracy = sum((predict == bad_labels).type(torch.FloatTensor)) / len(predict)
    print(''.join(['Have tested ', str(TEST_SIZE), ' bad pictures, accuracy = ', str(accuracy)]))
    print('complete testing')

