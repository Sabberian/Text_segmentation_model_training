import cv2
import os
from numpy import load as npload
from cfg import *
class Dataset():    
    def __init__(
            self, 
            images_dir, 
            masks_file, 
    ):
        self.ids = os.listdir(images_dir)
        self.images_paths = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.bin_masks = npload(masks_file)

    def __getitem__(self, i):
        image = cv2.imread(self.images_paths[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = self.bin_masks[self.ids[i]]
        mask = mask.astype('float')
        image = cv2.resize(image, (SIZE_X, SIZE_Y))
        mask = cv2.resize(mask, (SIZE_X, SIZE_Y))
        return image, mask
        
    def __len__(self):
        return len(self.ids)