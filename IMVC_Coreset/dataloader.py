import os
from pathlib import Path
import numpy as np
import scipy.io
from torch.utils.data import DataLoader, Dataset

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent / 'data'

def _standardize(values):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    safe_std = np.where(std == 0, 1.0, std)
    return (values - mean) / safe_std

class mfeat(Dataset):

    def __init__(self, path):
        self.OV1 = scipy.io.loadmat(path + 'mfeat.mat')['X1'].astype(np.float32)
        self.OV2 = scipy.io.loadmat(path + 'mfeat.mat')['X2'].astype(np.float32)
        self.OV3 = scipy.io.loadmat(path + 'mfeat.mat')['X3'].astype(np.float32)
        self.OV4 = scipy.io.loadmat(path + 'mfeat.mat')['X4'].astype(np.float32)
        self.OV5 = scipy.io.loadmat(path + 'mfeat.mat')['X5'].astype(np.float32)
        self.OV6 = scipy.io.loadmat(path + 'mfeat.mat')['X6'].astype(np.float32)
        self.V1 = _standardize(self.OV1)
        self.V2 = _standardize(self.OV2)
        self.V3 = _standardize(self.OV3)
        self.V4 = _standardize(self.OV4)
        self.V5 = _standardize(self.OV5)
        self.V6 = _standardize(self.OV6)
        num_classes = 10
        num_samples_per_class = 200
        self.L = np.repeat(np.arange(num_classes), num_samples_per_class)

    def __len__(self):
        return 2000

    def __getitem__(self, idx):
        return [self.V1[idx], self.V2[idx], self.V3[idx], self.V4[idx], self.V5[idx], self.V6[idx]]

class MSRCV1(Dataset):

    def __init__(self, path):
        file_path = path + 'MSRCV1_X.mat'
        self.OV1 = scipy.io.loadmat(path + 'MSRCV1_X.mat')['view_1'].astype(np.float32)
        self.OV2 = scipy.io.loadmat(path + 'MSRCV1_X.mat')['view_2'].astype(np.float32)
        self.OV3 = scipy.io.loadmat(path + 'MSRCV1_X.mat')['view_3'].astype(np.float32)
        self.OV4 = scipy.io.loadmat(path + 'MSRCV1_X.mat')['view_4'].astype(np.float32)
        self.OV5 = scipy.io.loadmat(path + 'MSRCV1_X.mat')['view_5'].astype(np.float32)
        self.OV6 = scipy.io.loadmat(path + 'MSRCV1_X.mat')['view_6'].astype(np.float32)
        self.L = scipy.io.loadmat(path + 'MSRCV1_X.mat')['Y'].astype(np.float32)
        self.L = self.L.flatten()
        self.L = self.L.tolist()
        self.L = [int(x) for x in self.L]
        num_classes = 7
        self.V1 = _standardize(self.OV1)
        self.V2 = _standardize(self.OV2)
        self.V3 = _standardize(self.OV3)
        self.V4 = _standardize(self.OV4)
        self.V5 = _standardize(self.OV5)
        self.V6 = _standardize(self.OV6)

    def __len__(self):
        return 210

    def __getitem__(self, idx):
        return [self.V1[idx], self.V2[idx], self.V3[idx], self.V4[idx], self.V5[idx], self.V6[idx]]

class Caltech5V(Dataset):

    def __init__(self, path):
        file_path = path + 'Caltech5V.mat'
        self.OV1 = scipy.io.loadmat(path + 'Caltech-5V.mat')['X1'].astype(np.float32)
        self.OV2 = scipy.io.loadmat(path + 'Caltech-5V.mat')['X2'].astype(np.float32)
        self.OV3 = scipy.io.loadmat(path + 'Caltech-5V.mat')['X3'].astype(np.float32)
        self.OV4 = scipy.io.loadmat(path + 'Caltech-5V.mat')['X4'].astype(np.float32)
        self.OV5 = scipy.io.loadmat(path + 'Caltech-5V.mat')['X5'].astype(np.float32)
        self.L = scipy.io.loadmat(path + 'Caltech-5V.mat')['Y'].astype(np.float32)
        self.L = self.L.flatten()
        self.L = self.L.tolist()
        num_classes = 7
        self.V1 = _standardize(self.OV1)
        self.V2 = _standardize(self.OV2)
        self.V3 = _standardize(self.OV3)
        self.V4 = _standardize(self.OV4)
        self.V5 = _standardize(self.OV5)

    def __len__(self):
        return 1400

    def __getitem__(self, idx):
        return [self.V1[idx], self.V2[idx], self.V3[idx], self.V4[idx], self.V5[idx]]

class Wikipedia(Dataset):

    def __init__(self, path):
        file_path = path + 'Wikipedia.mat'
        self.OV1 = scipy.io.loadmat(path + 'Wikipedia.mat')['view_1'].astype(np.float32)
        self.OV2 = scipy.io.loadmat(path + 'Wikipedia.mat')['view_2'].astype(np.float32)
        self.L = scipy.io.loadmat(path + 'Wikipedia.mat')['Y'].astype(np.float32)
        self.L = self.L.flatten()
        self.L = self.L.tolist()
        num_classes = 10
        self.V1 = _standardize(self.OV1)
        self.V2 = _standardize(self.OV2)

    def __len__(self):
        return 693

    def __getitem__(self, idx):
        return [self.OV1[idx], self.OV2[idx]]

class MNIST(Dataset):

    def __init__(self, path):
        file_path = path + 'MNIST_USPS.mat'
        self.OV1 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X1'].astype(np.float32)
        self.OV2 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X2'].astype(np.float32)
        self.OV1 = self.OV1.reshape((5000, -1))
        self.OV2 = self.OV2.reshape((5000, -1))
        self.L = scipy.io.loadmat(path + 'MNIST_USPS.mat')['Y'].astype(np.float32)
        self.L = self.L.flatten()
        self.L = self.L.tolist()
        num_classes = 10
        self.V1 = _standardize(self.OV1)
        self.V2 = _standardize(self.OV2)

    def __len__(self):
        return 5000

    def __getitem__(self, idx):
        return [self.OV1[idx], self.OV2[idx]]

def load_data(dataset, data_dir=None):
    data_path = Path(data_dir or os.environ.get('IMVC_DATA_DIR', DEFAULT_DATA_DIR)).expanduser().resolve()
    if not data_path.is_dir():
        raise FileNotFoundError(f'IMVC data directory does not exist: {data_path}')
    data_path = str(data_path) + os.sep
    if dataset == 'mfeat':
        dataset = mfeat(data_path)
        dims = [216, 76, 64, 6, 240, 47]
        view = 6
        data_size = 2000
        class_num = 10
        data_loader = DataLoader(dataset, batch_size=data_size, shuffle=False)
    elif dataset == 'MSRCV1':
        dataset = MSRCV1(data_path)
        dims = [1302, 48, 512, 100, 256, 210]
        view = 6
        data_size = 210
        class_num = 7
        data_loader = DataLoader(dataset, batch_size=data_size, shuffle=False)
    elif dataset == 'Caltech5V':
        dataset = Caltech5V(data_path)
        dims = [40, 254, 1984, 512, 928]
        view = 5
        data_size = 1400
        class_num = 7
        data_loader = DataLoader(dataset, batch_size=data_size, shuffle=False)
    elif dataset == 'Wikipedia':
        dataset = Wikipedia(data_path)
        dims = [128, 10]
        view = 2
        data_size = 693
        class_num = 10
        data_loader = DataLoader(dataset, batch_size=data_size, shuffle=False)
    elif dataset == 'MNIST_USPS':
        dataset = MNIST(data_path)
        dims = [784, 784]
        view = 2
        data_size = 5000
        class_num = 10
        data_loader = DataLoader(dataset, batch_size=data_size, shuffle=False)
    else:
        raise ValueError(f'Unknown dataset: {dataset}')
    return (dataset, dims, view, data_size, class_num, data_loader)
