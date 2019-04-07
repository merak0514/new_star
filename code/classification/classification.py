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
import numpy as np

b = '_b'
c = '_c'
end = '.jpg'
LR = 0.02
resnet18 = resnet.resnet18()
resnet18.train()

BATCH_SIZE = 64
EPOCH = 1
SAVE_ITER = 50
train_set_label_path = '../new_labels.csv'
train_set_path = '../../cut_data/'
save_path = './model/'
data = []
with open(train_set_label_path) as f:
    csv_reader = csv.reader(f)
    for row in csv_reader:
        data.append(row)
    data = data[1:]  # 去掉第一行
data = np.array(data)
print(len(data))
train_data = data[: int(0.7*len(data)), :]
test_data = data[int(0.7*len(data)):, :]
print('train_data', len(train_data))

# trainloader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)  # 生成一个个batch进行批训练，组成batch的时候顺序打乱取
# for i in trainloader:
#     print(i)
#     assert 0
classes = (0, 1)
criterion = nn.CrossEntropyLoss()  # 损失函数为交叉熵
optimizer = optim.SGD(resnet18.parameters(), lr=LR, momentum=0.9,
                      weight_decay=5e-4)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
state_dict_path = input('type the path of state_dict, or nothing to retrain the net')
if state_dict_path:
    model = torch.load(state_dict_path)
    optimizer.load_state_dict(model['optimizer_state_dict'])
    resnet18.load_state_dict(model['model_state_dict'])

if __name__ == '__main__':
    # print(resnet18)
    print('start training!')
    epoch_count = 0
    np.random.shuffle(train_data)
    epoch = 0
    while True:
        print('epoch:', epoch)
        resnet18.train()
        correct_sum = 0
        for batch_count in range(len(train_data) // BATCH_SIZE):
            data = train_data[batch_count*BATCH_SIZE: (batch_count+1)*BATCH_SIZE]
            images = torch.Tensor([])  # b,w,h,c
            labels = torch.zeros(BATCH_SIZE, dtype=torch.long)

            for i, datum in enumerate(data, 0):
                image_name = datum[0]
                # print(datum)
                label = int(datum[3])

                image_b = torch.Tensor(cv2.imread(''.join([train_set_path, image_name[:2], '/', image_name, b, end])))
                image_c = torch.Tensor(cv2.imread(''.join([train_set_path, image_name[:2], '/', image_name, c, end])))
                # image_b = torch.Tensor(cv2.imread('../../cut_data/d5/d52f52b895f03a214a3a077acd253066_0_b.jpg')[:, :, 0])
                # image_c = torch.Tensor(cv2.imread('../../cut_data/d5/d52f52b895f03a214a3a077acd253066_0_c.jpg')[:, :, 0])

                image_combine = torch.cat((image_b.unsqueeze(2), image_c.unsqueeze(2)), dim=2)  # 作为二通道的输入
                images = torch.cat((images, image_combine.unsqueeze(0)),0)
                labels[i] = label

                # print('labels', labels)
            images = images.permute((0, 3, 1, 2))
            # print('image shape', images.shape)

            optimizer.zero_grad()

            outputs = resnet18.forward(images)
            # outputs = outputs[0, :]
            # print('outputs shape', outputs.shape)
            # print('labels shape', labels.shape)
            # print(labels[2].type())
            # outputs = outputs[:, 1] > outputs[:, 0]
            # print(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            predict = (outputs [:, 1] > outputs[:, 0]).type(torch.LongTensor)
            correct_sum += sum((predict == labels).type(torch.FloatTensor))
            accuracy = sum((predict == labels).type(torch.FloatTensor)) / len(predict)
            print(''.join(['epoch: ', str(epoch), ', iter: ', str(batch_count),
                           ', loss: ', str(loss.item()), ', accuracy: ', str(accuracy)]))
            # print(correct_sum/((batch_count%SAVE_ITER + 1)*BATCH_SIZE))

            if batch_count % SAVE_ITER == 0:
                torch.save({
                    'epoch': epoch,
                    'batch': batch_count,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'model_state_diict': resnet18.state_dict(),
                    'accuracy': correct_sum/((batch_count%SAVE_ITER + 1)*BATCH_SIZE),

                }, ''.join([save_path, 'save_epoch_', str(epoch), '_batch_', str(batch_count), '.net']))
                print('save success')
                correct_sum = 0

        epoch += 1
