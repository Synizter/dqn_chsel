import numpy as np
from sklearn.preprocessing import MinMaxScaler
import scipy.io
import tensorflow as tf
from tqdm import tqdm

label_keys = {'ball':1,'box':2,'pen':3,'no':0}

def extract_data_from_key(key, mat_file, norm = True):
    temp = []
    for k in mat_file.keys():
        if k.find(key) != -1:
            temp.append(mat_file[k].T)
    data = np.vstack([t for t in temp])
    if norm:
        data = np.array([normalize(d) for d in data])
    label = np.ones((data.shape[0],), dtype=int) * label_keys[key]
    return data, label

def normalize(x:np.array):
#   norm = minmax_scale(x, axis = 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(x)
    return scaled

def get(fname):
    

    tx = []
    ty = []

    for f in fname:
        info = scipy.io.loadmat(f)
        
        ball_data, ball_label = extract_data_from_key('ball', info)
        box_data, box_label  = extract_data_from_key('box', info)
        pen_data, pen_label  = extract_data_from_key('pen', info)
        nm_data, nm_label  = extract_data_from_key('no', info)

        tx.append(np.vstack([ball_data, box_data, pen_data, nm_data]))
        ty.append(np.concatenate([ball_label, box_label, pen_label, nm_label]))

    X = np.vstack([d for d in tx])
    y = np.concatenate([d for d in ty])
    nbr_class = len(np.unique(y))
    y = tf.keras.utils.to_categorical(y,num_classes = nbr_class)
    return X, y


if __name__ == "__main__":
    fname = ['Datasets/Lai_JulyData.mat', 'Datasets/Suguro_JulyData.mat', 'Datasets/Takahashi_JulyData.mat']
    X, y = get(fname)
    print(X.shape, y.shape)
    
