import model
from model import *

import torch.utils.data as Utils

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from tqdm import tnrange
import scipy.ndimage
import cv2
import os
import csv
import pandas as pd
from IPython.display import clear_output
import imageio

import torch.optim as optim
import json

def import_json(path):
    with open(path, 'r') as file:
        hyper_parameters = json.load(file)
    return hyper_parameters

hyper_parameters = import_json('./hyper_parameters.json')
RESIZE = hyper_parameters['RESIZE']
BETA = hyper_parameters['BETA']

def create_csv():
    pokelist = os.listdir("./Pokemon/")
    with open("./PokeList", 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(pokelist)
        
        
# Dataloader class that works with VCTK datasets. Init require the path to a folder where voice_data are stored.
class PokeSet(Utils.Dataset): # Main dataset class for dataloader
    def __init__(self, path): # takes two input, the input and output Tensors
        self.path = path
        self.pokelist = pd.read_csv("PokeList",header=None)

    def __len__(self): # must be written for dataset module
        return len(os.listdir(self.path))
    
    def __getitem__(self,index): # must be written for dataset module
        pokemon = mpimg.imread(self.path + self.pokelist[index][0])
        if pokemon.shape[-1] == 4:
            pokemon=pokemon[:,:,:3]
        pokemon = cv2.resize(pokemon, (RESIZE,RESIZE))
        pokemon = (pokemon - np.min(pokemon))/np.max(pokemon - np.min(pokemon))
        return pokemon

    
def criterion(x_out, target, z_mean, z_logvar, alpha = 1, beta =  BETA):
    bce = F.mse_loss(x_out, target, size_average=False) #Use MSE loss for images
    kl = -0.5 * torch.sum(1 + z_logvar - (z_mean**2) - torch.exp(z_logvar)) #Analytical KL Divergence - Assumes p(z) is Gaussian Distribution
    loss = ((alpha * bce) + (beta * kl)) / x_out.size(0)    
    return loss, bce, kl


def load_checkpoint(filename):
    '''Loading function for the model before and during training
    From a checkpoint file, it loads and returns all necessary data (mode, optimiser, epoch number, losses)
    input: filename -> The name of the checkpoint file to be opened (.pth or .pt)
    output: net -> The saved model, including weights and biases
    output: epoch -> The epoch number at which the training was saved
    output: loss_save -> An array of all the saved batch losses during training
    output: optimizer -> The current state of the optimiser with its updated learning rates from training'''
    
    net = Net().cuda() # Initialize model  
    checkpoint = torch.load(filename) # load checkpoint data
    net.load_state_dict(checkpoint['state_dict'])
    epoch = checkpoint['epoch']
    losses = checkpoint['losses']
    optimizer = optim.Adam(net.parameters(), lr=0.0002)
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("Loaded checkpoint: " + filename)
    return net, epoch, losses, optimizer#, scheduler


def pokePlot(images, vae, ROW, COL, epoch):
    f, axarr = plt.subplots(ROW,COL, figsize=(25,ROW*4))
    for row in range(ROW//2):
        for col in range(COL):
            image = images[col+(COL*row),:,:,:].unsqueeze(0)
            axarr[2*row,col].imshow(image.squeeze().numpy())
            image = image.permute(0,3,1,2)
            x_out, z_mean, z_logvar = vae(Variable(image.float()).cuda())
            x_out = x_out.permute(0,2,3,1)
            axarr[2*row+1,col].imshow(x_out.data.cpu().squeeze().numpy())
    plt.show()
    f.savefig('./output/epoch_'+str(epoch)+'.png')