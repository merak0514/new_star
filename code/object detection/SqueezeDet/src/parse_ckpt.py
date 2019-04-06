# read the ckpt file
import os 
from tensorflow.python import pywrap_tensorflow 
import numpy as np

squeezeDet_chechpoint_path = './checkpoint/squeezeDet/model.ckpt-87000'
squeezeDetPlus_chechpoint_path = './checkpoint/squeezeDetPlus/model.ckpt-95000'
# read data from checkpoint file 
reader = pywrap_tensorflow.NewCheckpointReader(squeezeDetPlus_chechpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
# print tensor name and values 
for key in var_to_shape_map:
	print('tensor_name: {}'.format(key),end==',')
	param = reader.get_tensor(key)
	print('the shape is :{}'.format(np.shape(param)))