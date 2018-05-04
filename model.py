import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import json

def import_json(path):
    with open(path, 'r') as file:
        hyper_parameters = json.load(file)
    return hyper_parameters

hyper_parameters = import_json('./hyper_parameters.json')

LATENT_DIM = hyper_parameters['LATENT_DIM']
RESIZE = hyper_parameters['RESIZE']

class Net(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super(Net, self).__init__()
        
        #encoder
        self.conv_e1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
        self.bn_e1 = nn.BatchNorm2d(32)
        self.conv_e2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.bn_e2 = nn.BatchNorm2d(64)
        self.conv_e3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn_e3 = nn.BatchNorm2d(128)
        self.conv_e4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn_e4 = nn.BatchNorm2d(256)
        
        self.fc_mean = nn.Linear(256*RESIZE//16*RESIZE//16, latent_dim)
        self.fc_logvar = nn.Linear(256*RESIZE//16*RESIZE//16, latent_dim)
        
        #decoder
        self.fc_d = nn.Linear(latent_dim, 256*RESIZE//16*RESIZE//16)
        
        self.up = nn.Upsample(scale_factor=2)
        
        self.conv_d1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.bn_d1 = nn.BatchNorm2d(128)
        self.conv_d2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn_d2 = nn.BatchNorm2d(64)
        self.conv_d3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.bn_d3 = nn.BatchNorm2d(32)
        self.conv_d4 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
        
    def encoder(self, x):
        ''' encoder: q(z|x)
            input: x, output: mean, logvar
        '''
        x = F.leaky_relu(self.bn_e1(self.conv_e1(x)))
        x = F.leaky_relu(self.bn_e2(self.conv_e2(x)))
        x = F.leaky_relu(self.bn_e3(self.conv_e3(x)))
        x = F.leaky_relu(self.bn_e4(self.conv_e4(x)))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        z_mean = self.fc_mean(x)
        z_logvar = self.fc_logvar(x)
        return z_mean, z_logvar
    
    def latent(self, z_mu, z_logvar):
        ''' 
            encoder: z = mu + sd * e
            input: mean, logvar. output: z
        '''
        sd = torch.exp(z_logvar * 0.5)
        e = Variable(torch.randn(sd.size())).cuda()
        z = e.mul(sd).add_(z_mu)
        return z 
    
    def decoder(self, z):
        '''
            decoder: p(x|z)
            input: z. output: x
        '''
        x = self.fc_d(z)
        x = x.view(-1, 256, RESIZE//16, RESIZE//16)
        x = F.leaky_relu(self.bn_d1(self.conv_d1(self.up(x))))
        x = F.leaky_relu(self.bn_d2(self.conv_d2(self.up(x))))
        x = F.leaky_relu(self.bn_d3(self.conv_d3(self.up(x))))
        x = F.sigmoid(self.conv_d4(self.up(x)))
        return x.view(-1,3,RESIZE,RESIZE)

    def forward(self, x):
        z_mean, z_logvar = self.encoder(x)
        z = self.latent(z_mean, z_logvar)
        x_out = self.decoder(z)
        return x_out, z_mean, z_logvar