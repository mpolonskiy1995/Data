import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import torchmetrics


from random import randrange
import os
import pandas as pd
import numpy as np

import cv2 as cv
from PIL import Image
import PIL.ImageOps    
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
from tqdm import tqdm


dev = "cuda:0" if torch.cuda.is_available() else "cpu" 
device = torch.device(dev) 
size = (150,150)
path = os.path.abspath(os.getcwd())
imgs_mean = 0.9898
imgs_std = 0.0786

class HazelNet(nn.Module):
    """Class for instanciating the NN

    Args:
        nn (nn.Module): super class to inherit from in pytorch
    """

    def __init__(self):
        """ Cosntructor for initialization
        """
        super(HazelNet, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
  
        # over-write the first conv layer to be able to read images
        # as resnet18 reads (3,x,x) where 3 is RGB channels
        # whereas MNIST has (1,x,x) where 1 is a gray-scale channel
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.fc_in_features = self.resnet.fc.in_features 
        # remove the last layer of resnet18 (linear layer which is before avgpool layer)
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))

        # add linear layers to compare between the features of the two images
        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            nn.Linear(1024, 128),
            )        
       
        # initialize the weights
        self.resnet.apply(self.init_weights)
        self.fc.apply(self.init_weights)   
        
    def init_weights(self, m):
        """Function for weight init

        Args:
            m (module): module to use for init
        """
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)
    
    def forward_once(self, inputs):
        """Helper function for forward path

        Args:
            inputs (tensor): input tensor

        Returns:
            tensor: output tensor
        """
        output = self.resnet(inputs)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output
    
    def distance_layer(self, vec1, vec2):
        """Function for calculating the cosine similarity between two tensors

        Args:
            vec1 (tensor): tensor for template images
            vec2 (tensor): tensor for images to compare with

        Returns:
            tensor: tensor containing the calculated similarity as float
        """
        cos = torch.nn.CosineSimilarity()
        similarity = cos(vec1, vec2) 
        return similarity

    def forward(self, template, img):
        """Main function for forward path

        Args:
            template (tensor): tensor of template images
            img (tensor): tensor of images to compare

        Returns:
            tensor: tensor containing the calculated similarity as float
        """
        output1 = self.forward_once(template)
        output2 = self.forward_once(img)
        output = self.distance_layer(output1,output2)
 
        return output

transform = transforms.Compose([

    transforms.Grayscale(),
    # resize
    transforms.Resize(size),
    # to-tensor
    transforms.ToTensor(),
    # normalize
    transforms.Normalize((0.98), (0.07))
])

def readImg_url ( url1, url2, isblack = False, plot = False):

    """Function for reading images into processable tensors. Can draw a picture of processed images

    Args:
        url1 (string): url to template image
        url2 (string): url to image for comparison
        isblack (bool, optional): inverts image colors if set to true. Defaults to False.
        plot (bool, optional): plots imported images if set to true. Defaults to False.

    Returns:
        tensor, tensor: two tensors containing the processed images ready for prediction
    """

    realim1 = Image.open(url1)
    realim2 = Image.open(url2)
    # invert white and black if image is on white background
    if isblack:
        realim1 = PIL.ImageOps.invert(realim1)
        realim2 = PIL.ImageOps.invert(realim2)

    template = transform(realim1).unsqueeze(0).to(device)
    img = transform(realim2).unsqueeze(0).to(device)

    if plot:
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(realim1)
        ax[1].imshow(realim2)
    return template, img        

def predict (model, template, img):
    """Function for predicting the correlation score between two images

    Args:
        model (HazelNet): The model to use for prediction
        template (tensor): template image
        img (tensor): image to compare to

    Returns:
        float: label containing predicted similarity score
    """
    model.eval()
    with torch.no_grad():
        ypred = model(template,img)
    return round(ypred.item(),2)