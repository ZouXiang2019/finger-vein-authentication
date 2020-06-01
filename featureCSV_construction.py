import numpy as np
import matplotlib.pyplot as plt
import csv
import scipy.io as scio
from array import array
import torch.nn as nn
import torch
from cmath import pi
import trace


path = r'E:\data_ori\1.csv'

EPC = ['11','12','13','14','15','16','17',
       '21','22','23','24','25','26','27',
       '31','32','33','34','35','36','37',
       '41','42','43','44','45','46','47',
       '51','52','53','54','55','56','57',
       '61','62','63','64','65','66','67',
       '71','72','73','74','75','76','77'
       ]
time = {}

RSS = {}

phase = {}


def max_length(dic):
    max_len = 0
    for key in dic.keys():
        if len(dic[key]) > max_len:
            max_len = len(dic[key])
    return max_len     

def min_length(dic):
    min_len = 10000000
    for key in dic.keys():
        if len(dic[key]) < min_len:
            min_len = len(dic[key])
    return min_len 

def interpolate_list(dic,max_len):#插值
    new_dic = dic
    for key in dic:
        value_list = dic[key]
        value_list = torch.from_numpy(np.array(value_list)).view(1,1,-1)
        value_list = list(np.array((nn.functional.interpolate(value_list,size=max_len, mode='linear', align_corners=None)).view(-1)))
        new_dic[key] = value_list
    return new_dic

def value_array_form(dic,max_len):#EPC矩阵
    value_array = []
    for i in range(max_len):
        tmp = [] 
        for id_ in EPC:
            EPC_name = 'E0' + id_
            tmp.append(dic[EPC_name][i])
        value_array.append(np.array(tmp))
    return np.array(value_array)

def value_time_array_form(value_dic,time_dic,min_len):#rss矩阵、time矩阵
    value_array = []
    time_array = []
    for i in range(min_len):
        RSS_tmp = [] 
        time_tmp = []
        for id_ in EPC:
            EPC_name = 'E0' + id_
            RSS_tmp.append(value_dic[EPC_name][i])
            time_tmp.append(time_dic[EPC_name][i])
        

        value_array.append(np.array(RSS_tmp))
        time_array.append(np.array(time_tmp))
        
     
    return np.array(value_array), np.array(time_array)

def gradient_calculation(value_array,time_array):#计算梯度
    gradient_list = []
    i = 0
    
    while i+2 < len(value_array):
        tmp = []
        for j in range(len(value_array[0])):
            gradient = (value_array[i+2][j] - value_array[i][j])/(time_array[i+2][j] - time_array[i][j])#RSS/Time
#             gradient = (value_array[i+2][j]-value_array[i+1][j])/(time_array[i+2][j] - time_array[i+1][j]) + (value_array[i+1][j]-value_array[i][j])/(time_array[i+1][j] - time_array[i][j])
            tmp.append(gradient)
        gradient_list.append(np.array(tmp))
        i += 1
    return np.array(gradient_list)
            
def trace_calculation(RSS_gradient_array,time_array):#计算位置
    trace = []
    for i in range(len(RSS_gradient_array)):
        
        for j in range(len(RSS_gradient_array[0])):
            candidate_EPC_idx  = [] 
            
            if abs(RSS_gradient_array[i][j]) > 0.032:
                candidate_EPC_idx.append(j)
                
            if len(candidate_EPC_idx) > 0:
                earliest_EPC_idx = candidate_EPC_idx[0]
                earliest_time = time_array[i][earliest_EPC_idx]
                
                for EPC_idx in candidate_EPC_idx:
                    if time_array[i][EPC_idx] < earliest_time:
                        earliest_EPC_idx = EPC_idx
                        earliest_time = time_array[i][earliest_EPC_idx]
                EPC_name = 'E0' + EPC[j]
                if EPC_name not in trace:
                    trace.append(EPC_name)                    
    return trace     

def feature_formalization(RSS_array,phase_array):  #规整化

    feature = []
    i = 0
    while i < len(RSS_array):
        j = 0
        tmp_rss = []
        tmp_phase = []
        tmp = []
        
       
        while(j<10 and j < len(RSS_array)):   
         
         
            tmp_rss.append(np.array(RSS_array[j]))          
            tmp_phase.append(np.array(phase_array[j]))
            j+=1
           
        i+=10
     
        if len(tmp_rss) == 10:
        
            tmp.append(np.array(tmp_rss))
            tmp.append(np.array(tmp_phase))
            feature.append(np.array(tmp))


    return np.array(feature)


tmp = []

csv_content = csv.reader(open(path,'r'))

for line in csv_content:
    tmp.append(line)
#print(tmp[])

csv_content = tmp
line_count = len(csv_content)
#print(line_count)

for id_ in EPC:

    EPC_name = 'E0' + id_
    time_tmp = []
    RSS_tmp = []
    phase_tmp = []
    i = 1
    while i < line_count:
        if csv_content[i][0] == EPC_name :
            time_tmp.append(i) 
            RSS_tmp.append(float(csv_content[i][6]))
            phase_tmp.append(float(csv_content[i][7]))
        i += 1 
#     print(len(RSS_tmp))


    time[EPC_name] = time_tmp
    RSS[EPC_name] = RSS_tmp
    phase[EPC_name] = phase_tmp
 
# max_len = max_length(RSS)
min_len = min_length(RSS)
# print(max_len)
#RSS = interpolate_list(RSS,max_len)
# RSS_array  = value_array_form(RSS,max_len)

RSS_array,time_array = value_time_array_form(RSS,time, min_len)
phase_array,time_array = value_time_array_form(phase,time, min_len)
feature_array = feature_formalization(RSS_array,phase_array)

print(feature_array.shape)


np.save(r'D:\Program Files\JetBrains\PyCharm Community Edition 2018.3.4\CNN-1\1.npy',feature_array)