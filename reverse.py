import csv
import os
import cv2


def save_img(path, filename, img, text):
    """
    保存图片，文件名命名格式：地址 + 文件名 + 修改方式 + 后缀
    """
    save_path = path+'\\' + \
        os.path.splitext(filename)[0]+text+'.jpg'
    if not os.path.exists(path):
        os.makedirs(path)
    cv2.imwrite(save_path, img)  # change this with manipulation!!

    print(save_path)


def flip_img_hori(img, x, y):
    """
    水平翻转
    """
    img1 = img.copy()
    max_rows = img.shape[1]
    # max_cols = img.shape[0]
    img1 = cv2.flip(img1, 1, dst=None)  # horizontally
    x1 = max_rows - x
    y1 = y
    return img1, x1, y1


def flip_img_verti(img, x, y):
    """
    竖直翻转
    """
    img1 = img.copy()
    # max_rows = img.shape[1]
    max_cols = img.shape[0]
    img1 = cv2.flip(img1, 0, dst=None)  # vertically
    x1 = x
    y1 = max_cols - y
    return img1, x1, y1


def rotate_img(img, x, y, degree):
    """
    任意角度旋转图片，num是需要旋转后并剪出的图片的大小
    """
    img1 = img.copy()
    max_rows = img.shape[1]
    max_cols = img.shape[0]
    Matrix = cv2.getRotationMatrix2D((max_rows/2, max_cols/2), degree, 1)  # rotate matrix
    img1 = cv2.warpAffine(img, Matrix, (max_rows, max_cols))
    return img1, x, y


def import_data(label_path):
    """
    从csv文件中导入数据
    """
    train_data = []

    with open(label_path) as csvfile:
        csv_reader = csv.reader(csvfile)
        # data_header = next(csv_reader)

        for row in csv_reader:
            # if row[3] == 1:
            train_data.append(row)

    # 创建写入的文件
    if not os.path.exists('../good_data/good_list.csv'):
        with open('../good_data/good_list.csv', 'w', newline='') as csvfile:  # 为write_data 做准备
            csv_writer = csv.writer(csvfile)
            csv_writer = csv_writer.writerow(
                ["filename", "x", "y", "label"])

    return train_data 



if __name__ == "__main__":
    
    train_image_path = '../cut_data/'
    train_label_path = '../cut_data/new_labels.csv'
    save_path = '../good_data/'

    train_data = import_data(train_label_path)
    # print(train_data)

    for row in train_data:

        if row[3] == '1' :
            print(row )
            img_name, x, y, label = row[0], int(row[1]), int(row[2]), row[3]

            img_path = train_image_path + img_name + '.jpg'
            print(img_path)

            img = cv2.imread(img_path)
            
            text = '_flip_hori'
            img1, x1, y1 = flip_img_hori(img, x, y)
            save_img(save_path, img_name, img1, text)

            text = '_flip_verti'
            img2, x2, y2 = flip_img_verti(img, x, y)
            save_img(save_path, img_name, img2, text)
            

            text = '_rotate_90'
            img3, x3, y3 = rotate_img(img, x, y, 90)
            save_img(save_path, img_name, img3, text)

            text = '_rotate_270'
            img3, x3, y3 = rotate_img(img, x, y, 90)
            save_img(save_path, img_name, img3, text)

            text = '_rotate_180'
            img3, x3, y3 = rotate_img(img, x, y, 90)
            save_img(save_path, img_name, img3, text)







   


    
