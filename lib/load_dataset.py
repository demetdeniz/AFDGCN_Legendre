import os
import numpy as np

def load_st_dataset(dataset):
    # output B, N, D
    if dataset == 'PEMS04':
        data_path = os.path.join('/content/AFDGCN_Garnoldi/data/PEMS04/PEMS04.npz')
        data = np.load(data_path)['data'][:, :, :1]  # only the first dimension, traffic flow data
    elif dataset == 'PEMS08':
        data_path = os.path.join('/content/AFDGCN_Garnoldi/data/PEMS08/PEMS08.npz')
        data = np.load(data_path)['data'][:, :, :1]  # only the first dimension, traffic flow data
    elif dataset == 'PEMS03':
        data_path = os.path.join('/content/AFDGCN_Garnoldi/data/PEMS03/PEMS03.npz')
        data = np.load(data_path)['data'][:, :, :1]  # only the first dimension, traffic flow data
    elif dataset == 'PEMS07':
        data_path = os.path.join('/content/AFDGCN_Garnoldi/data/PEMS07/PEMS07.npz')
        data = np.load(data_path)['data'][:, :, :1]  # only the first dimension, traffic flow data
    elif dataset == 'Kcetas':
        data_path = os.path.join('/content/AFDGCN_Garnoldi/data/Kcetas/morning_only.npz')
        data = np.load(data_path)['data'][:, :, :1] 
    elif dataset == 'Konya':
        data_path = os.path.join('/content/AFDGCN_Garnoldi/data/Konya/konya_kavşak.npz')
        data = np.load(data_path)['data'][:, :, :1]  # only the first dimension, traffic flow data
    else:
        raise ValueError
    print('Load %s Dataset shaped: ' % dataset, data.shape)
    return data

# 测试内容
# data = load_st_dataset(dataset = 'PEMS08')
# print(data.shape)
