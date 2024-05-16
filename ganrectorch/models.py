import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self, img_h, img_w, conv_num, conv_size, dropout, output_num):
        super(Generator, self).__init__()
        units = 128
        fc_size = img_w ** 2

        self.fc_stack = nn.Sequential(
            self.dense_norm(units, dropout),
            self.dense_norm(units, dropout),
            self.dense_norm(units, dropout),
            self.dense_norm(fc_size, 0),
        )

        self.conv_stack = nn.Sequential(
            self.conv2d_norm(conv_num, conv_size+2, 1),
            self.conv2d_norm(conv_num, conv_size+2, 1),
            self.conv2d_norm(conv_num, conv_size, 1),
        )

        self.dconv_stack = nn.Sequential(
            self.dconv2d_norm(conv_num, conv_size+2, 1),
            self.dconv2d_norm(conv_num, conv_size+2, 1),
            self.dconv2d_norm(conv_num, conv_size, 1),
        )

        self.last = self.conv2d_norm(output_num, 3, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_stack(x)
        x = x.view(-1, 1, img_w, img_w)  # Reshape
        x = self.conv_stack(x)
        x = self.dconv_stack(x)
        x = self.last(x)
        return x

    def dense_norm(self, units, dropout):
        return nn.Sequential(
            nn.Linear(units, units),
            nn.BatchNorm1d(units),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def conv2d_norm(self, out_channels, kernel_size, stride):
        return nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size, stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def dconv2d_norm(self, out_channels, kernel_size, stride):
        return nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size, stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )


import torch
from torch import nn

class Discriminator(nn.Module):
    def __init__(self, nang, px):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 16, (5, 5), stride=(2, 2), padding='same'),
            nn.Conv2d(16, 16, (5, 5), stride=(1, 1), padding='same'),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),

            nn.Conv2d(16, 32, (5, 5), stride=(2, 2), padding='same'),
            nn.Conv2d(32, 32, (5, 5), stride=(1, 1), padding='same'),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),

            nn.Conv2d(32, 64, (3, 3), stride=(2, 2), padding='same'),
            nn.Conv2d(64, 64, (3, 3), stride=(1, 1), padding='same'),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),

            nn.Conv2d(64, 128, (3, 3), stride=(2, 2), padding='same'),
            nn.Conv2d(128, 128, (3, 3), stride=(1, 1), padding='same'),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),

            nn.Flatten(),
            nn.Linear(nang * px * 128, 256),
            nn.Linear(256, 128),
        )

    def forward(self, input):
        return self.main(input)