import torch

from torch.utils.data import Dataset, DataLoader
import numpy as np

class TrainTestDataset():
    # it is preferable to shuffle the data everytime the training is initialized
    # since the shuffle is done randomly sometimes it might be possible that
    # not enough shapes appear in training dataset
    def __init__(self, PATH):
        # -- load dataset (assumes datasets are saved "certain_dir/dataset" and file named dataset_x.pt or dataset_y.pt
        full_ds_x = torch.load(PATH + "_x.pt")
        full_ds_y = torch.load(PATH + "_y.pt")
        # -- normalize data between 0 and 1 (optional)

        # get dataset size
        ds_size = full_ds_x.shape[0] # x and y have the same size
        split_num = 6
        test_size = ds_size // split_num

        # generate random indices
        idx = np.random.permutation(ds_size)

        # split training and testing data
        X_test = full_ds_x[:test_size]
        X_train = full_ds_x[test_size:]

        Y_test = full_ds_y[:test_size]
        Y_train = full_ds_y[test_size:]

        # create training and testing data loaders
        self.train_dataloader = TrainDataloader(X_train, Y_train)
        self.test_dataloader = TestDataloader(X_test, Y_test)

class TrainDataloader(Dataset):
    def __init__(self, x_data, y_data):
        self.x = x_data
        self.y = y_data
        self.n_samples = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index] # x = y in encoder decoder
    
    def __len__(self):
        return self.n_samples

class TestDataloader(Dataset):
    def __init__(self, x_data, y_data):
        self.x = x_data 
        self.y = y_data 
        self.n_samples = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index] # x = y in encoder decoder

    def __len__(self):
        return self.n_samples

if __name__=='__main__':
    PATH = "../dataset/output/dataset"
    # initialize train-test dataset
    tt_dataset = TrainTestDataset(PATH)
    # Train
    train_data = tt_dataset.train_dataloader
    print(f'train size: ', train_data.__len__())

    # Test
    test_data = tt_dataset.test_dataloader
    print(f'test size: ', test_data.__len__())

    print(f'dataset shape: ', test_data.x.shape) 
