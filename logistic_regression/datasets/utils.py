import numpy as np
import sklearn
from sklearn.datasets import load_svmlight_file, fetch_rcv1


def get_dataset(dataset, data_path='./datasets/'):
    if dataset in ['covtype', 'real-sim', 'webspam', 'YearPredictionMSD']:
        return load_svmlight_file(data_path + dataset + '.bz2')
    elif dataset in ['mushrooms', 'a5a']:
        return load_svmlight_file(data_path + dataset)
    elif dataset == 'rcv1':
        return fetch_rcv1(data_home=data_path, return_X_y=True)
    elif dataset in ['heart_scale', 'w8a']:
        return load_svmlight_file(data_path + dataset + '.txt')
    elif dataset == 'gisette':
        return load_svmlight_file(data_path + dataset + '_scale.bz2')
    elif dataset == 'YearPredictionMSD_binary':
        A, b = load_svmlight_file(data_path + dataset + '.bz2')
        b = b > 2000
        return A, b
    elif dataset == 'rcv1_binary':
        A, b = fetch_rcv1(return_X_y=True)
        freq = np.asarray(b.sum(axis=0)).squeeze()
        main_class = np.argmax(freq)
        b = (b[:, main_class] == 1) * 1.
        b = b.toarray().squeeze()
        return A, b