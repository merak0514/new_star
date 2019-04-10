# !/user/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 16:54
# @Author   : Merak
# @File     : test2.py
# @Software : PyCharm
import csv
train_set_label_path = './new_labels.csv'

train_set_path = '../cur_data/'
train_data = []
with open(train_set_label_path) as f:
    csv_reader = csv.reader(f)
    for row in csv_reader:
        train_data.append(row)
    train_data = train_data[1:]  # 去掉第一行

print(type(train_data[0][3]))
