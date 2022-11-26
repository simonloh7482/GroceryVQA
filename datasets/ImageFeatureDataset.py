import os

import h5py
import torch
import torch.utils.data as data


class ImageFeatureDataset(data.Dataset):

    def __init__(self, feat_path):
       
        self.h5_path = feat_path

        # make sure the features are extracted and stored in .h5 file before proceeding
        assert os.path.isfile(self.h5_path), \
            'Features .h5 file not found in {}'.format(self.h5_path)

        self.h5_file = h5py.File(self.h5_path, 'r')
        self.img_features = self.h5_file['att'] 

    def __getitem__(self, index):
        return torch.from_numpy(self.img_features[index].astype('float32'))

    def __len__(self):
        return self.img_features.shape[0]
