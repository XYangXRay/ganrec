import torch
import torch.nn as nn
import torch.nn.init as init
import pandas as pd
from ganrectorch.utils import to_device
# from ganrectorch.ganrec_dataloader import *


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
   
    def forward(self, x):
        return x.view(x.size(0), -1)
      
class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)
    
class Transpose(nn.Module):
    def __init__(self):
        super(Transpose, self).__init__()

    def forward(self, x):
        return x.view(-1, 1)

class Generator(nn.Module):
    def __init__(self, img_h, img_w, conv_num, conv_size, dropout, output_num):
        super(Generator, self).__init__()
        units_out = 128
        self.img_w = img_w
        self.img_h = img_h

        # Calculate the size after the fully connected layers
        input_size = img_w * img_h
        fc_size = img_w**2

        self.fc_stack = nn.ModuleList([
            self.dense_norm(input_size, units_out, dropout),
            self.dense_norm(units_out, units_out, dropout),
            self.dense_norm(units_out, units_out, dropout),
            self.dense_norm(units_out, fc_size, 0)
        ])

        self.conv_stack = nn.ModuleList([
            self.conv2d_norm(1, conv_num, conv_size+2, 1),
            self.conv2d_norm(conv_num, conv_num, conv_size+2, 1),
            self.conv2d_norm(conv_num, conv_num, conv_size, 1),
        ])

        self.dconv_stack = nn.ModuleList([
            self.dconv2d_norm(conv_num, conv_num, conv_size+2, 1),
            self.dconv2d_norm(conv_num, conv_num, conv_size+2, 1),
            self.dconv2d_norm(conv_num, conv_num, conv_size, 1),
        ])

        self.last = self.conv2d_norm(conv_num, output_num, 3, 1)
        
        self.generator_model = to_device(nn.Sequential(
            Flatten(),
            Transpose(),
            *self.fc_stack,
            Reshape((-1, 1, self.img_w, self.img_h)),
            *self.conv_stack,
            *self.dconv_stack,
            self.last)
                                          )
        
        
    def forward(self, x):
        self.pred = self.generator_model(x)
        return self.pred
    
        
    
        # self= to_device(self)

    # def forward(self, x):
    #     x = x.view(x.size(0), -1)  # Flatten
    #     x = self.fc_stack(x)
    #     x = x.view(-1, 1, self.img_w, self.img_w)  # Reshape to (batch_size, channels, height, width)
    #     x = self.conv_stack(x)
    #     x = self.dconv_stack(x)
    #     x = self.last(x)
    #     return x

    def dense_norm(self, units_in, units_out, dropout):
        return nn.Sequential(
            nn.Linear(units_in, units_out),
            # nn.BatchNorm1d(units_out),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def conv2d_norm(self, in_channels, out_channels, kernel_size, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, 
                      out_channels, 
                      kernel_size, stride, 
                      padding='same',),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def dconv2d_norm(self, in_channels, out_channels, kernel_size, stride):
        padding = (kernel_size - 1) // 2
        output_padding = stride-1
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, 
                               out_channels, 
                               kernel_size, 
                               stride, 
                               padding=padding, 
                               output_padding=output_padding),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )



class Discriminator(nn.Module):
    def __init__(self, nang, px):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 16, (5, 5), stride=(2, 2)),
            nn.Conv2d(16, 16, (5, 5), stride=(1, 1)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),

            nn.Conv2d(16, 32, (5, 5), stride=(2, 2)),
            nn.Conv2d(32, 32, (5, 5), stride=(1, 1)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),

            nn.Conv2d(32, 64, (3, 3), stride=(2, 2)),
            nn.Conv2d(64, 64, (3, 3), stride=(1, 1)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),

            nn.Conv2d(64, 128, (3, 3), stride=(2, 2)),
            nn.Conv2d(128, 128, (3, 3), stride=(1, 1)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),

            nn.Flatten(),
            # nn.Linear(nang * px * 128, 256),
            # nn.Linear(256, 128),
        )
        
        self = to_device(self)

    def forward(self, input):
        return self.main(input)