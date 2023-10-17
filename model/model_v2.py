import torch
# from torch.utils.data import DataLoader
import torch.nn as nn
# from torch.autograd import Variable
# import matplotlib.pyplot as plt
# import torchvision
# from dataloader import getDataset
# import os
# import numpy as np
# from tqdm import tqdm
# import math



class HousePriceModel_CNN(nn.Module):
    def __init__(self, input_dim):
        super(HousePriceModel_CNN, self).__init__()
        self.input_dim = input_dim
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = self.input_dim, out_channels = 32, kernel_size = 3, stride = 1, padding = 1,),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 9, stride = 1, padding = 6),# stride = 1, padding = (kernel_size-1)/2 = (5-1)/2
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 9, stride = 1, padding = 5,),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3,3), stride = 1, padding = 1,),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels = 64, out_channels =32, kernel_size = (1,1), stride = 1, padding = 0,),
            nn.LeakyReLU(0.1),
            # nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 5, stride = 1, padding = 0,),
            # nn.LeakyReLU(0.1),
            # nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (3,3), stride = 1, padding = 0,),
            # nn.LeakyReLU(0.1),
            # nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = (1,1), stride = 1, padding = 0,),
            # nn.LeakyReLU(0.1),
            # nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (3,3), stride = 1, padding = 0,),
            # nn.LeakyReLU(0.1),
            # nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = (1,1), stride = 1, padding = 0,),
            # nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels = 32, out_channels = 16, kernel_size = (1,1), stride = 1, padding = 0,),
            nn.Conv2d(in_channels = 16, out_channels = 1, kernel_size = (1,1), stride = 1, padding = 0,),
        )
        # self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = torch.reshape(x,(-1,self.input_dim,1,1))
        x = self.conv1(x)
        x = torch.reshape(x,(-1,1,1,1))
        # print(x.shape)

        return x