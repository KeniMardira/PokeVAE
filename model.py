import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import matplotlib.pyplot as plt

def import_json(path):
    with open(path, 'r') as file:
        hyper_parameters = json.load(file)
    return hyper_parameters

hyper_parameters = import_json('./hyper_parameters.json')

LATENT_DIM = hyper_parameters['LATENT_DIM']
RESIZE = hyper_parameters['RESIZE']

class PlanarFlow(nn.Module):
    def __init__(self, K=8, latent_dim=LATENT_DIM):
        super(PlanarFlow, self).__init__()
        self.wb = []
        self.u = []
        for i in range(K):
            self.wb.append(nn.Linear(latent_dim, 1))
            self.u.append(nn.Linear(1, latent_dim, bias=False))
            # self.u[i].weight.data = (self.u_constraint(self.u[i].weight.data, self.wb[i].weight.data))
            
    def scalar_func(self, x):
        m_x = -1 + F.softplus(x)
        return m_x
    
    def u_constraint(self, u, w):
        u_out = u + ((self.scalar_func(w @ u)) - (w @ u))*(w.transpose(0,1)/torch.norm(w))
        return u_out
        
    def transform(self, z, u, wb):
        out = z + u(F.tanh(wb(z)))
        return out
    
    def determinant(self, z, u, w):
        phi = (1/(torch.cosh(w(z)))) @ w.weight
        ln_det = torch.log(torch.abs(1 + u(phi)))
        return ln_det 
        
    def forward(self, z):
        z_T = z
        for idx in range(len(self.wb)):
            z_T = self.transform(z_T, self.u[idx], self.wb[idx])
        return z_T
        

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
        
        ##### VAMP PRIOR STUFF ####
        self.number_components = 500 #num of pseudo-inputs
        
        self.meanproj = nn.Linear(self.number_components, (3*RESIZE*RESIZE), bias=False)
        self.nonlinear = nn.Hardtanh(min_val=0.0, max_val=1.0)
        
        self.idle_input = Variable(torch.eye(self.number_components, self.number_components), requires_grad = False).cuda()
        
        self.means = nn.Sequential(self.meanproj, self.nonlinear)
        ##### VAMP PRIOR STUFF ####
        
        
    def calculate_loss(self, x, beta=10.):
        x_out, z_mean, z_logvar = self.forward(x)

        z = self.latent(z_mean, z_logvar)
        log_p_z = self.log_p_z(z)   # b
        # z = b x L
        # z_mean = b x L
        # z_logvar = b x L
        log_q_z = self.log_norm(z, z_mean, z_logvar, dim=1) # b
        RE = F.l1_loss(x_out, x, size_average=False)
        KL = torch.sum(-(log_p_z - log_q_z))
        
        loss = RE + beta*KL
        
        
        return loss, RE, KL

    
    def log_p_z(self, z):
        C = self.number_components # number of pseudo inputs
        
        X = self.means(self.idle_input) # get C amount of pseudo inputs
        X = X.view(-1, 3, RESIZE, RESIZE) # reshape pseudo inputs to the same shape as the actual input
        z_p_mean, z_p_logvar = self.encoder(X) # grab the mean and logvar of the aggregated posterior (actual prior modeled by pseudo input)
        
        z_expand = z.unsqueeze(1) # b x 1 x L
        means = z_p_mean.unsqueeze(0) # 1 x pseudo x L
        logvars = z_p_logvar.unsqueeze(0) # 1 x pseudo x L
        a = self.log_norm(z_expand, means, logvars, dim=2) - math.log(C) # b x pseudo
        a_max, _ = torch.max(a,1) # b
        
        log_prior = a_max + torch.log(torch.sum(torch.exp(a-a_max.unsqueeze(1)),1)) # b
        return log_prior
    
    
    
    def log_norm(self, z, zmean, zlogvar, dim, average=False):
        log_normal = -0.5 * ( zlogvar + torch.pow( z - zmean, 2 ) / torch.exp( zlogvar ) )
        if average:
            return torch.mean( log_normal, dim )
        else:
            return torch.sum( log_normal, dim )
        
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