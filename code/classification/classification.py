# !/user/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 16:59
# @Author   : Merak
# @File     : classification.py
# @Software : PyCharm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import resnet
import csv
import cv2

b = '_b'
c = '_c'
end = '.jpg'
LR = 0.1
resnet18 = resnet.resnet18()
resnet18.train()

BATCH_SIZE = 32
EPOCH = 1
train_set_label_path = '../new_labels.csv'
train_set_path = '../../cur_data/'
train_data = []
with open(train_set_label_path) as f:
    csv_reader = csv.reader(f)
    for row in csv_reader:
        train_data.append(row)
    train_data = train_data[1:]  # 去掉第一行
# print(train_data)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)  # 生成一个个batch进行批训练，组成batch的时候顺序打乱取

classes = (0, 1)
criterion = nn.CrossEntropyLoss()  # 损失函数为交叉熵，多用于多分类问题
optimizer = optim.SGD(resnet18.parameters(), lr=LR, momentum=0.9,
                      weight_decay=5e-4)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）

if __name__ == '__main__':
    print(resnet18)
    print('start training!')
    epoch_count = 0
    for epoch in range(EPOCH):
        print('epoch:', epoch)
        resnet18.train()
        for i, data in enumerate(trainloader, 0):
            images = []
            labels = []
            for datum in data:
                image_name = datum[0]
                label = int(datum[3])

                image_b = torch.Tensor(cv2.imread(''.join([train_set_path, image_name[:2], '/', image_name, b, end])))
                image_c = torch.Tensor(cv2.imread(''.join([train_set_path, image_name[:2], '/', image_name, c, end])))
                image_combine = torch.cat((image_b.unsqueeze(2), image_c.unsqueeze(2)), dim=2)  # 作为二通道的输入
                images.append(image_combine)
                labels.append(label)

            optimizer.zero_grad()

            outputs = resnet18(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print(loss.item())
