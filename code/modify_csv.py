# coding=utf-8
# run in python3
import csv

init_label_path = '../cut_data/labels.csv'
new_label_path = '../cut_data/new_labels.csv'
csv_title = ['id','x','y','classification']

new_csv = open(new_label_path,'w+', newline='')
csv_write = csv.writer(new_csv,delimiter=',')
csv_write.writerow(csv_title)

with open(init_label_path) as csvfile:
    csv_reader = csv.reader(csvfile)
    data_header = next(csv_reader)
    for row in csv_reader:
    	id_str = row[0]
    	if(row[1]=='0'):
    		cls = 0
    		x_coor = '0'
    		y_coor = '0'
    	else:
    		cls = 1
    		coor = row[1].strip('()')
    		[x_coor,y_coor] = coor.split(',')
    	line_data = [id_str,x_coor,y_coor,cls]
    	csv_write.writerow(line_data)
    	print('%s is rewrite over......'%id_str)

    	


