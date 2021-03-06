# !/user/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 19:56
# @Author   : Merak
# @File     : classification2.py.py
# @Software : PyCharm
import classification
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import resnet
import csv
import cv2
import numpy as np
import os
import re

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
epoch = 10 
LR = 0.01
BATCH_SIZE = 64
SAVE_ITER = 500
train_data_set_path = '../../good_data2/'
IMAGE_B = 'image_b/'
IMAGE_C = 'image_c/'
model_path = './model25/'
b = '_b'
c = '_c'
end = '.jpg'
end2 = '.png'


def import_data():
    train_set_label_path = train_data_set_path + 'labels.csv'
    bad_data = []
    good_data = []
    with open(train_set_label_path) as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            if row[1] == '0':
                bad_data.append(row[0])
            else:
                good_data.append(row[0])
    bad_data = np.array(bad_data)
    good_data = np.array(good_data)
    print('bad_data', len(bad_data))
    print('good_data', len(good_data))
    bad_train_data_ = bad_data[: int(0.8 * len(bad_data))]
    bad_test_data = bad_data[int(0.8 * len(bad_data)):]
    good_train_data_ = good_data[: int(0.8 * len(good_data))]
    good_test_data = good_data[int(0.8 * len(good_data)):]
    return bad_train_data_, bad_test_data, good_train_data_, good_test_data


if __name__ == '__main__':
    resnet18 = resnet.resnet18().cuda()
    resnet18.train()
    classes = (0, 1)
    criterion = nn.CrossEntropyLoss()  # 损失函数为交叉熵
    optimizer = optim.SGD(resnet18.parameters(), lr=LR, momentum=0.9,
                          weight_decay=5e-4)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）

    state_dict_path = classification.find_newest_model(model_path_=model_path)
    if state_dict_path:
        model = torch.load(model_path + state_dict_path)
        optimizer.load_state_dict(model['optimizer_state_dict'])
        resnet18.load_state_dict(model['model_state_dict'])

    print('start training!')
    bad_train_data, _, good_train_data, _= import_data()
    resnet18.train()
    while True:  # 循环：一个一个epoch训练
        print('epoch:', epoch)
        np.random.shuffle(bad_train_data)  # 每过一个epoch：洗牌
        np.random.shuffle(good_train_data)
        correct_sum = 0
        batch_count = 0
        while True:  # 循环：一个一个batch训练
            bad_data_counter = 0
            good_data_counter = 0
            # 构建一个0/1随机分布（不一定数量相等，但数学期望上数量相等）的label list
            labels = torch.ge(torch.randn(BATCH_SIZE), torch.randn(BATCH_SIZE)).cuda().type(torch.LongTensor)

            images = torch.Tensor([]).cuda()
            for ty in labels:  # 按照构建的label list 挑选正负样本
                if ty == 0:  # 负样本
                    image_name = bad_train_data[bad_data_counter]
                    image_b = torch.Tensor(
                        cv2.imread(''.join([train_data_set_path, image_name[:2], '/', image_name, b, end2])))[:, :, 0]
                    image_c = torch.Tensor(
                        cv2.imread(''.join([train_data_set_path, image_name[:2], '/', image_name, c, end2])))[:, :, 0]

                    image_combine = torch.cat((image_b.unsqueeze(2), image_c.unsqueeze(2)), dim=2).cuda()  # 作为二通道的输入
                    images = torch.cat((images, image_combine.unsqueeze(0)), 0)
                    bad_data_counter += 1
                    if bad_data_counter >= len(bad_train_data):
                        break
                elif ty == 1:  # 正样本
                    image_name = good_train_data[good_data_counter]
                    image_b = torch.Tensor(
                        cv2.imread(''.join([train_data_set_path, image_name[:2], '/', image_name, b, end2])))[:, :, 0]
                    image_c = torch.Tensor(
                        cv2.imread(''.join([train_data_set_path, image_name[:2], '/', image_name, c, end2])))[:, :, 0]

                    image_combine = torch.cat((image_b.unsqueeze(2), image_c.unsqueeze(2)), dim=2).cuda()  # 作为二通道的输入
                    images = torch.cat((images, image_combine.unsqueeze(0)), 0)
                    good_data_counter += 1
                    if good_data_counter >= len(good_train_data):
                        break
            if len(images) < BATCH_SIZE:
                break
            images = images.permute((0, 3, 1, 2))  # 换为b, c, w, h

            optimizer.zero_grad()
            outputs = resnet18.forward(images).cuda()
            loss = criterion(outputs, labels.cuda())
            loss.backward()
            optimizer.step()
            batch_count += 1

            predict = (outputs[:, 1] > outputs[:, 0]).cuda().type(torch.LongTensor)
            correct_sum += sum((predict == labels).type(torch.FloatTensor))
            accuracy = sum((predict == labels).type(torch.FloatTensor)) / len(predict)
            if batch_count % 100 == 0:  # print sth every 100 iter
                print(''.join(['epoch: ', str(epoch), ', batch: ', str(batch_count),
                               ', loss: ', str(loss.item()), ', accuracy: ', str(accuracy)]))

            if batch_count % SAVE_ITER == 0:  # save the model
                torch.save({
                    'epoch': epoch,
                    'batch': batch_count,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'model_state_dict': resnet18.state_dict(),
                    'accuracy': correct_sum / (SAVE_ITER * BATCH_SIZE),
                }, ''.join([model_path, 'save_epoch_', str(epoch), '_batch_', str(batch_count), '.net']))
                print('save success, ', 'accuracy: ', str(correct_sum / (SAVE_ITER * BATCH_SIZE)))
                correct_sum = 0
                classification.delete_former_model(model_path)

        epoch += 1
