import argparse
import time

import h5py
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.models as models
import yaml
from tqdm import tqdm
from torchvision.models import ResNet152_Weights

import sys, os

cwd = os.getcwd()
sys.path.append(cwd)

from datasets.ImageDataset import ImageDataset

# the module to extract image features from ImageDataset
class ImageFeatureExtractor(nn.Module):

    def __init__(self):
        super(ImageFeatureExtractor, self).__init__()
        # pretrained resnet152 is used
        self.model = models.resnet152(weights="DEFAULT")

        # Save attention features (tensor)
        def save_att_features(module, input, output):
            self.att_feat = output

        # forward hook that executes after forward is called
        # store attention features from the output of layer 4 in resnet 152
        self.model.layer4.register_forward_hook(save_att_features)

    def forward(self, x):
        self.model(x)
        return self.att_feat  # [batch_size, 2048, 14, 14]


def main():
    # Load config yaml file
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_config', default='config/default.yaml', type=str,
                        help='path to a yaml config file')
    args = parser.parse_args()

    if args.path_config is not None:
        with open(args.path_config, 'r') as handle:
            config = yaml.safe_load(handle)
         
    # store paths to all images into a dataset (train, test, val)
    # and perform resize, center-crop, to tensor, normalize transformation on all images
    dataset = ImageDataset(config)

    # load images in batches
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['images']['batch_size'],
        num_workers=config['images']['data_workers'],
        shuffle=False
    )

    # the shape of attention features extracted after layer 4 in resnet 152
    att_features_shape = (
        len(data_loader.dataset),
        config['images']['output_feat'],
        config['images']['output_size'],
        config['images']['output_size']
    )

    # create h5 file to store extracted attention features
    h5_file = h5py.File(config['images']['feat_path'], 'w')
    h5_att = h5_file.create_dataset('att', shape=att_features_shape, dtype='float16')

    # store image names for order of extraction
    dt = h5py.special_dtype(vlen=str)
    h5_img_names = h5_file.create_dataset('img_name', shape=(len(data_loader.dataset),), dtype=dt)
    
    # load the image feature extractor class
    net = ImageFeatureExtractor()
    # eval mode
    net.eval()
    
    if torch.cuda.is_available():
        # transfer to GPU if available
        net = net.cuda()
        # Benchmark mode is good whenever your input sizes for your network do not vary
        # cudnn will look for the optimal set of algorithms for that particular configuration. 
        # usually leads to faster runtime
        # NOTE: input size is set in config['img_size']
        cudnn.benchmark = True
    
    # keep track of the image batches to store in h5 file
    idx = 0
    batch_size = config['images']['batch_size']
    
    print('Extracting features ...')
    begin = time.time()
        
    # start extracting features
    # no calculating gradient
    with torch.no_grad():
        for i, inputs in enumerate(tqdm(data_loader)):
            # retrieve the image 
            inputs_img = inputs['img']
            if torch.cuda.is_available():
                # transfer to GPU if available
                inputs_img = inputs_img.cuda()
 
            # passing image to network and features are extracted 
            att_feat = net(inputs_img)

            # features are stored into h5 file
            h5_att[idx:idx + batch_size, :, :] = att_feat.data.cpu().numpy().astype('float16')
            # image name are stored into h5 file to keep track of the order
            h5_img_names[idx:idx + batch_size] = inputs['name']

            idx += batch_size
            
    h5_file.close()

    end = time.time() - begin

    print('Image feature extraction time taken: ')
    print('{}m and {}s'.format(int(end / 60), int(end % 60)))
    print('Created file : ' + config['images']['feat_path'])


if __name__ == '__main__':
    main()
    
