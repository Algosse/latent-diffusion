import torch
from torchvision.transforms import transforms
import os
import numpy as np
from PIL import Image
import random

class SIAR(torch.utils.data.Dataset):
    """
    Dataset class for SIAR dataset
    
    The dataset is constitued of N folders, N is the number of images of the dataset
    Each folder contains 11 images, gt.png is the ground truth image, the others are distorted images (from 1 to 10)
    """
    SPLIT_FILE = "split.csv"
    WRONG_FILE = "wrongs.txt"
    
    def __init__(self, root, set_type='train', transform=None, generate_split=False, resolution=64, max_sequence_size=10):
        """
        Args:   
            path (str): path to the dataset
            set (str): train, val or test
            transform (torchvision.transforms): transform to apply to the images
            generate_split (bool): if True, generate a new split file
        """
        
        assert set_type in ['train', 'val', 'test'], "set_type must be train, val or test"
        
        self.root = root
        
        if not self._split_exists():
            if generate_split:
                self._generate_split()
            else:
                raise RuntimeError("Split file does not exist, please generate it with generate_split=True")

        self.wrong_images = self._get_wrong_images_list()

        self.set_type = set_type
        self.split = self._load_split()
        self.images = self._load_images()
        
        self.len = len(self.images)

        self.transform = transform
        
        self.resolution = resolution
        
        self.max_sequence_size = max_sequence_size
        
    def __getitem__(self, index):
        """
        Args:
            index (int or slice): index of item
        returns:
            dict: {'gt': gt, 'input': inputs} if single index
            list: [{'gt': gt, 'input': inputs}, ...] if slice
        """
        if isinstance(index, slice):
            return [self._getitem(i) for i in range(*index.indices(self.len))]
        else:
            return self._getitem(index)
        
    def _getitem(self, index):
        """ Get an item
        Args:
            index (int): index of item
        Returns:
            dict: {'gt': gt, 'input': inputs}
                gt: torch.tensor
                input: torch.tensor
                    Shape is determined by the transform applied. Without transform, shapes are gt:
                    (256, 256, 3), input: (10, 256, 256, 3)
        """

        if index >= self.len:
            raise IndexError("Index out of range")
        if index < 0:
            index += self.__len__()
        
        # read grund truth image
        gt = Image.open(os.path.join(self.root, self.images[index], "gt.png"))
        gt = gt.resize((self.resolution, self.resolution))
        
        # read input images
        input = []
        for i in range(1, self.max_sequence_size + 1):
            im = Image.open(os.path.join(self.root, self.images[index], str(i) + ".png"))
            im = im.resize((self.resolution, self.resolution))
            input.append(im)
        
        # apply transform if any
        if self.transform:
            gt = self.transform(gt)
            input = [self.transform(im) for im in input]
            
            input = torch.stack(input)
        else:
            """ to_tensor = transforms.ToTensor()
            gt = to_tensor(gt)
            input = [to_tensor(im) for im in input]
            
            input = torch.stack(input) """
            
            gt = np.array(gt).astype(np.uint8)
            input = [np.array(im).astype(np.uint8) for im in input]
            
            input = np.stack(input)
        
        return {
            'data': (gt/127.5 - 1).astype(np.float32), 
            'label': (input/127.5 - 1).astype(np.float32),
            'name': self.images[index]
        }
        
    def __len__(self):
        """ Get the length of the dataset """
        return self.len
    
    def _split_exists(self):
        return os.path.exists(os.path.join(self.root, self.SPLIT_FILE))
    
    def _wrong_exists(self):
        return os.path.exists(os.path.join(self.root, self.WRONG_FILE))
    
    def _generate_split(self):
        """ Generate a split for the dataset 
            If the split already exists, please delete the file before generating a new one
        """
        print("Generating split...")
        images = os.listdir(self.root)
        
        split = []
        count = [0, 0, 0]
    
        for im in images:
            r = random.random()
            if r <= 0.8:
                split.append(im + ',train')
                count[0] += 1
            elif r <= 0.9:
                split.append(im + ',val')
                count[1] += 1
            else:
                split.append(im + ',test')
                count[2] += 1
                
        with open(os.path.join(self.root, self.SPLIT_FILE), 'w') as f:
            f.write('\n'.join(split))
            
        print("Split generated")
        print("Train: {}, Val: {}, Test: {}".format(count[0], count[1], count[2]))

    def _load_split(self):
        """ Load the split file """
        
        with open(os.path.join(self.root, self.SPLIT_FILE), 'r') as f:
            data = f.read().split("\n")
        
        split = {}
        for line in data:
            if line == '':
                continue
            name, set_type = line.split(',')
            split[name] = set_type
            
        return split
    
    def _load_images(self):
        """ Load the images according to the split """
        im = []
        
        for name, set_type in self.split.items():
            if set_type == self.set_type and name not in self.wrong_images:
                im.append(name)
                
        return im
    
    def _get_wrong_images_list(self):
        """ Some images are misslabelled in the dataset, this function returns the list of those images """
        
        if self._wrong_exists():
            with open(os.path.join(self.root, self.WRONG_FILE), 'r') as f:
                data = f.read().split("\n")
                print("Wrong images excluded")
                return data

        print("No wrong images file found")
        return None
        
if __name__ == "__main__":
    
    transform = transforms.Compose(
        [transforms.ToTensor()])

    data = SIAR("data/SIARmini", set_type='train', transform=transform)
    
    dataloader = torch.utils.data.DataLoader(data, batch_size=4)
    
    im = next(iter(dataloader))

    print(data[0]['data'].size())
    
    print(len(data))
    