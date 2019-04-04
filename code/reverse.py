import csv
import os
import cv2


def import_data(label_path):
    """
    从csv文件中导入数据
    """

    train_data = []
    with open(label_path) as csvfile:
        csv_reader = csv.reader(csvfile)
        # data_header = next(csv_reader)

        for row in csv_reader:
            train_data.append(row)

    # # 创建写入的文件
    # if not os.path.exists('list.csv'):
    #     with open('list.csv', 'w', newline='') as csvfile:  # 为write_data 做准备
    #         csv_writer = csv.writer(csvfile)
    #         csv_writer = csv_writer.writerow(
    #             ["index", "filename", "x", "y", "label"])

    # return train_data, info



if __name__ == "__main__":
    
    train_image_path = '../cut_data/'
    train_label_path = '../cut_data/new_labels.csv'
    save_path = '../reversed_data/'

    train_data, info = import_data(train_label_path)
    name, x, y, label = info[0], int(info[1]), int(info[2]), info[3]

    

    crop_and_save('./new', info, img_list, img_path, nums=60)
