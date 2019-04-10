# !/user/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import os
import cv2
import csv
b = '_b'
c = '_c'
end = '.jpg'


def cut_black(img1, img2):
    black_count = [0, 0, 0, 0]
    for direction in range(4):
        img1 = np.rot90(img1, direction)
        img2 = np.rot90(img2, direction)
        flag = 0
        size = img1.shape
        for i in range(size[0]):
            for j in range(size[1]):
                if img1[i, j] != 0:
                    flag = 1
                    break
            if flag == 1:
                break
            black_count[direction] += 1
        img1 = img1[black_count[direction]:, :]
        img2 = img2[black_count[direction]:, :]
        img1 = np.rot90(img1, -direction)
        img2 = np.rot90(img2, -direction)
    return img1, img2


def cut(image1, image2, cut_size=(100, 100)):
    size = image1.shape
    xs = np.arange(size[0] // cut_size[0]) * cut_size[0]
    xs = np.append(xs, size[0] - cut_size[0])
    ys = np.arange(size[1] // cut_size[1]) * cut_size[14]
    ys = np.append(ys, size[1] - cut_size[1])
    cuts_pos = []
    for i in xs:
        for j in ys:
            cuts_pos.append((i, j))
    cuts_b = []
    cuts_c = []
    for position in cuts_pos:
        image = image1[position[0]: position[0] + size[0], position[1]: position[1] + size[1]]
        cuts_b.append(image)
    for position in cuts_pos:
        image = image2[position[0]: position[0] + size[0], position[1]: position[1] + size[1]]
        cuts_c.append(image)
    return cuts_b, cuts_c


def compute_diff(image1, image2):
    """计算差值,小于0归零"""
    image = np.array(image1, np.int) - np.array(image2, np.int)
    image *= (image > 0)
    image = np.array(image, np.uint8)
    return image


def cut_too_small(image, lambda_=1.5):
    """去掉过小的点（噪声）"""
    aver = np.sum(image)/np.sum(image != 0)
    image *= (image > aver * lambda_)
    return image


def cut_too_large(image):
    """超过一定阈值的点设为255: 避免出现星星变亮导致插值图值很大"""
    image_ = np.array((image < 150) * image, dtype=np.uint8)
    temp = np.array((image > 150) * 255, dtype=np.uint8)
    image_ += temp
    return image_


def middle_filter(image):
    """中值滤波: 把九个点的中位数作为中间点的值"""
    x_len = np.size(image, 0)
    y_len = np.size(image, 1)
    for i in range(x_len - 2):
        for j in range(y_len - 2):
            # if np.sum(image[i:i+3, j:j+3]) == image[i+1, j+1]:
            #     image[i+1, j+1] = 0
            image[i + 1, j + 1] = np.sort(np.hstack(image[i:i + 3, j:j + 3]))[4]

    return image.copy()  # .copy有毒


def adjust_average(image1, image2):
    """
    adjust the average of image1 to fit image2
    note: only count non-zero
    :param image1:
    :type image1: np.array
    :param image2:
    :type image2: np.array
    :return: new image1
    :rtype: np.array
    """
    av1 = np.sum(image1)/np.sum(image1 != 0)
    av2 = np.sum(image2)/np.sum(image2 != 0)
    image1 = np.array(image1, np.float32)
    scale = av2 / av1
    # print(scale)
    image1 *= scale
    image1 = np.array(np.rint(image1), np.uint8)
    return image1.copy()


def random_cut(image_origin_1, image_origin_2, size, choosing_length, least_gap, origin_pos, ignore_anomaly=True):
    """
    把两张图片同时切为指定大小的区块，每个区块在每张图中的位置相同
    :type image_origin_1: np.array
    :param size: (x, y) 切后的大小
    :param choosing_length: 可供选择的区域长度，每个x的值都会在此区域中选择；但越大的选择空间意味着越少的图片
    :param least_gap: 最小的间隔：两个选取点之间的x或y坐标的最小差值。
    :param origin_pos: 初始标注的位置
    :param ignore_anomaly: 忽略异常
    :return: 两个list，分别为两张图切完之后的合集；
    """
    image_origin_1 = np.array(image_origin_1, np.uint8)
    x_len, y_len = np.shape(image_origin_1)
    # x_len, y_len = (1000, 100)  # 测试用
    if size[0] >= x_len or size[1] >= y_len:
        print("too large size")
        if not ignore_anomaly:
            input("type any key to continue")
        return -1
    x_num = (x_len - size[0] + least_gap)//(choosing_length + least_gap)
    y_num = (y_len - size[0] + least_gap)//(choosing_length + least_gap)
    x_sections = [[i*choosing_length + i*least_gap, (i+1)*choosing_length + i*least_gap] for i in range(x_num)]
    y_sections = [[i*choosing_length + i*least_gap, (i+1)*choosing_length + i*least_gap] for i in range(y_num)]
    x_choices = [np.random.randint(j[0], j[1]) for j in x_sections]
    y_choices = [np.random.randint(j[0], j[1]) for j in y_sections]

    combines = []  # 所有开始切的坐标的合集
    labels = []  # 和combines一一对应，为每一个坐标所对应图片中
    for i in x_choices:
        for j in y_choices:
            combines.append((i, j))  # 相当于添加每个方框的开始坐标
            if (i <= origin_pos[0] < i+size[0]) and (j <= origin_pos[1] < j+size[1]):
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


def _process_and_cut_a_image(image_name, pos, csv_file, train_image_path='../af2019-cv-training-20190312/'):
    """处理并切一张图片, pos为原图中标注的位置"""
    img_b = cv2.imread(''.join((train_image_path, image_name[:2], '/', image_name, b, end)))
    img_b = np.array(img_b[:, :, 0], np.uint8)
    img_c = cv2.imread(''.join((train_image_path, image_name[:2], '/', image_name, c, end)))
    img_c = np.array(img_c[:, :, 0], np.uint8)

    img_b, img_c = cut_black(img_b, img_c)
    img_b = adjust_average(img_b, img_c)
    img_b = cut_too_small(img_b, lambda_=1.4)
    img_b = cut_too_large(img_b)
    img_b = middle_filter(img_b)
    img_c = cut_too_small(img_c, lambda_=1.4)
    img_c = cut_too_large(img_c)
    img_c = middle_filter(img_c)

    temp = random_cut(img_b, img_c, (50, 50), 1, 50, pos)
    if temp == -1:
        return
    else:
        cut_images_b, cut_images_c, labels = temp

    path = ''.join(['../cut_data2/', image_name[:2]])
    if not os.path.exists(path):
        print(path)
        os.makedirs(path)
    for i in range(len(cut_images_b)):
        image = cut_images_b[i]
        im_path = ''.join([path, '/', image_name, '_', str(i), b, end])
        cv2.imwrite(im_path, image)
        data_row = [image_name+'_'+str(i), labels[i]]
        csv.writer(csv_file).writerow(data_row)

    for i in range(len(cut_images_c)):
        image = cut_images_c[i]
        im_path = ''.join([path, '/', image_name,  '_', str(i), c, end])
        cv2.imwrite(im_path, image)


def process_and_cut_all_image(csv_path='../af2019-cv-training-20190312/list.csv', start=0):
    if not os.path.exists(csv_path):
        print('No file')
        return -1

    train_data = []
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            train_data.append(row)
    train_data = train_data[1+start:]  # 去掉第一行
    print('The length of train_data is {}'.format(len(train_data)))

    if input('Print y to continue') is not 'y':
        exit()

    csv_file = open('../cut_data2/labels.csv', 'a+')
    # csv_file = open('../cut_data2/labels.csv', 'a+', newline='')
    for datum in train_data:
        print(datum)
        image_name = datum[0]
        pos = (int(datum[1]), int(datum[2]))
        _process_and_cut_a_image(image_name, pos, csv_file)
        # break

    csv_file.close()


if __name__ == '__main__':
    # random_cut([[0]], [[0]], (50, 50), 20, 10, [50, 60])
    # 危险!
    process_and_cut_all_image(start=0)
    pass
