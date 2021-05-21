from PIL import Image
import torch
import copy

class ReadPILImage():
    """ PyTorch transform to read color image from an image path. """

    def __init__(self):
        pass

    def __call__(self, img_path):
        # Read as PIL iamge
        img = Image.open(img_path).convert('RGB')
        return img
    
############### new forms 

#t3 = class_labels.index(phase)
#torch.Tensor(t3).type(torch.int64)

import numpy as np
from typing import List


class LabelToInt():
    def __init__(self, fields: List[str], labels: List[str]):
        self.fields = fields
        self.labels = labels
        #self.class_labels = ['Preparation', 'CalotTriangleDissection', 'ClippingCutting', 'GallbladderDissection', \
                #'GallbladderPackaging', 'CleaningCoagulation', 'GallbladderRetraction']

        self.tv_tform = labels.index

    def __call__(self, data: dict) -> dict:
        for field in self.fields:
            data[field] = self.labels.index(data[field])

        return data
    

class LabelToTensor():
    def __init__(self, fields: List[str]):
        self.fields = fields   

    def __call__(self, data: dict) -> dict:
        for field in self.fields:
            data[field] = torch.FloatTensor([data[field]])#.type(torch.float) #.unsqueeze(1)#type(torch.float) 
            #data[field] = torch.Tensor([data[field]]).type(torch.int32)                           
        return data
    
    

class DicttoImageandLabel():
    def __init__(self,image: List[str], label: List[str]):
        self.image = image 
        self.label = label 

    def __call__(self, data: dict) -> tuple:
            imgs = data['frame_path']
            lbls = data['label']
            return (imgs, lbls)