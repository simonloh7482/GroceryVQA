import os

import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image



class ImageDataset(data.Dataset):

    def __init__(self, conf):
        self.path = conf['images']['img_dir']
        self.transform = self.get_transform(conf['images']['img_size']);

        # common image file format supported by PIL 
        IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
        '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']

        # Load the PATHS to the images available in the folder                
        self.image_names = []
        for filename in os.listdir(self.path):
            if any(filename.endswith(extension) for extension in IMG_EXTENSIONS):
                self.image_names.append(filename)
                
        if len(self.image_names) == 0:
            raise (RuntimeError("Found 0 images in " + self.path + "\n"
                "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))
        else:
            print('Found {} images in {}'.format(len(self), self.path))


    def __getitem__(self, index):
        item = {}
        item['name'] = self.image_names[index]
        item['path'] = os.path.join(self.path, item['name'])

        # Use PIL to load the image
        item['img'] = Image.open(item['path']).convert('RGB')
        if self.transform is not None:
            item['img'] = self.transform(item['img'])

        return item

    def __len__(self):
        return len(self.image_names)


    def get_transform(self, img_size):
        # a pipeline of a series of transformation
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            # ImageNet normalization setting is used
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
            ])
