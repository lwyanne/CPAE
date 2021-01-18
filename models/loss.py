import os,sys
import numpy as np
import pandas as pd
from torch import nn
import torch
top_path=os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(top_path) 

global mapping_list
global mapping_length
global convert_M
global convert_ts 
# this is a 17 element long list 
# each sub-list is the positon of encoded variable 


def get_mapping_info(mapping_path=os.path.join(top_path,'ref/mapping_channel.csv')):
    """
    read mapping_channel csv and extract the info about how variables are encoded
    """
    mapping = pd.read_csv(mapping_path,index_col=0)
    mapping_array = mapping.iloc[:,1:].values
    mapping_list = [list(range(*iilist)) for iilist in mapping_array]  # creat a series range from begin to end posi
    mapping_length = [len(iilist) for iilist in mapping_list]          # number of columns for each variables

    start_end_index = np.array([np.sum(mapping_length[:i]) for i in range(len(mapping_length))] + [59]).astype(int)

    convert_M = np.zeros((17,76)).astype(int)

    for i in range(17):
        convert_M[i,range(start_end_index[i],start_end_index[i+1])] = 1

    return mapping_list,mapping_length,convert_M

mapping_list,mapping_length,convert_M = get_mapping_info()
convert_ts = torch.tensor(convert_M)


def mask_where(x,true_only = False):
    """
    the mapping function return the position where mask == 1
    """
    try:
        assert(isinstance(x,np.ndarray))
    except:
        x = x.detach().cpu().numpy()
    
    assert(x.shape[1] == 76)
    mask = x[:,-17:]
    if true_only:
        i,j = np.where(mask==1)     # generally, Boolen is `mask == 1`
        return i,j+59
    
    else:
        return [slice(None,None),slice(59,x.shape[1])]

def mapping_where(x,mapping_list=mapping_list,mapping_length = mapping_length):
    """
    the mapping function return the position where mask == 1
    """
    try:
        assert(isinstance(x,np.ndarray))
    except:
        x = x.detach().cpu().numpy()
        
    assert(x.shape[1] == 76)
    
    mask = x[:,-17:]
    if np.sum(mask == 1) == 0:
        return None,None

    test_array = list(zip(*np.where(mask==1)))     # generally, Boolen is `mask == 1`

    broadcast_ls = []
    # mapping ...
    for i,j in test_array:
        broadcast_i = [i]*mapping_length[j]       # one variable may mapped to several columns 
        broadcast_j = mapping_list[j]        
        broadcast_ls += list(zip(broadcast_i,broadcast_j))
    ii,jj = zip(*broadcast_ls)
    return ii,jj

def Chimera_loss(D,X,nce,Lambda=[5,1,2,2],detail_mse=False,**kwarg):
    """

    :param D:
    :param X:
    :param nce:
    :param Lambda: weights for [nce,MSE ,mask_mse ,mapping_mse]
    :param detail_mse:
    :param kwarg:
    :return:
    """

    if X.shape[0]==1: X=X.squeeze(0)
    Lambda = torch.tensor(Lambda).float().cuda()
    Lambda = Lambda / sum(Lambda) * 10
    MSE = 0;
    mask_mse = 0;
    mapping_mse = 0
    D = D.transpose(1, 2)
    batch_size =  X.shape[0]
    for i in range(X.shape[0]):  # iter within batch
        x = X[i, :, :]
        d = D[i, :, :]
        # print(x.shape,d.shape)
        if d.shape[0]==76:
            d=d.transpose(0,1)
            x=x[:d.shape[0],:]



        if x.transpose(0, 1).shape == d.shape:
            d = d.transpose(0, 1)
        # print(x.shape,d.shape)

        assert (x.shape == d.shape)  # just check
        assert ((len(x.shape) == 2) & (x.shape[1] == 76))
        pair_distance = torch.pow(x - d, 2)

        MSE += torch.mean(pair_distance) / batch_size # total MSE of the sample
        mask_mse += torch.mean(pair_distance[mask_where(x, **kwarg)]) / batch_size # the MSE of only mask code
        mapping_mse += torch.mean(pair_distance[mapping_where(x)]) / batch_size  # the MSE of

    losses = torch.stack([nce, MSE, mask_mse, mapping_mse])
    if detail_mse:  # design for function record_loss
        return losses
    else:
        return torch.dot(Lambda, losses)


def record_loss(D,X,nce):
    """
    return in the order of ["NCE","MSE","MASK_MSE","MAPPING_MSE"]
    """
    with torch.no_grad():
        loss_ls = [loss.item() for loss in Chimera_loss(D,X,nce,detail_mse=True)]
    return loss_ls


def mask_mapping_M(x):
    """
    return a mask matrix and mapping matrix, for the matrix computation of loss
        the input matrix x : shape (batch_size,76)
    """
    # the starting position of each variable, len = 17, sum = 59
    start_end_index = [np.sum(mapping_length[:i]) for i in range(len(mapping_length))] + [59] #to define the end
    start_end_index = np.array(start_end_index).astype(int)        # index should be int
    
    # try:
    #     assert(isinstance(x,np.ndarray))
    #     x0 = x
    # except:
    #     x0 = x.detach().cpu().numpy()    # copy the x to avoid auto grad error
    
    assert(x.shape[1] == 76)
    
    mask_matrix = x.detach()
    mask_matrix[:,:-17] = 0
    
    mask = x[:,-17:].detach()
    
    mapping_matrix = torch.mm(mask,convert_ts.to(mask.device).float())  # retrieveing true value by matrix multiply
                
    return mask_matrix,mapping_matrix