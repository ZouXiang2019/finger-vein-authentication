from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import csv
import scipy.io as scio
from array import array
import torch.nn as nn
import torch
from cmath import pi
import trace
tu = "\左小拇指"
filePath = r'E:\finger vein\data\savaPicture'+tu+'RIO.bmp'#请改变图片
I = Image.open(filePath, 'r')
# I.show()
# I.save('./save.png')
I_array = np.array(I)
print (I_array.shape)
# pic = I_array.reshape(532,784)
# print(I_array)
print(len(I_array))
print(len(I_array[0]))
new_array = []  # type:
for i in range(532):
    RSS_tmp = []
    # for j in range(len(I_array[0])):
    #     RSS_tmp.append(I_array[i][j])
    # new_pic = I_array(i).reshape(28,28)
    new_array.append(np.array(I_array))
print(len(new_array[3]))
np.save(r'D:\Program Files\JetBrains\PyCharm Community Edition 2018.3.4\CNN-1'+tu+'_lab_40'
        r'.npy', new_array)