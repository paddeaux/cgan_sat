import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, classes, d=512):
        super(Generator, self).__init__()
        self.deconv1_1 = nn.ConvTranspose2d(100, d, 4, 1, 0)
        self.deconv1_1_bn = nn.BatchNorm2d(d)
        self.deconv1_2 = nn.ConvTranspose2d(classes, d, 4, 1, 0)
        self.deconv1_2_bn = nn.BatchNorm2d(d)
        self.deconv2 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d)
        self.deconv3 = nn.ConvTranspose2d(d, d//2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d//2)
        self.deconv4 = nn.ConvTranspose2d(d//2, d//4, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d//4)
        
        # Additional layers for 128x128
        self.deconv5 = nn.ConvTranspose2d(d//4, d//8, 4, 2, 1)
        self.deconv5_bn = nn.BatchNorm2d(d//8)

        # Additional layers for 256x256
        self.deconv6 = nn.ConvTranspose2d(d//8, d//16, 4, 2, 1)
        self.deconv6_bn = nn.BatchNorm2d(d//16)

        # Final layer for 256x256
        self.final = nn.ConvTranspose2d(d//16, 3, 4, 2, 1)


    def forward(self, input, label):
        x = F.leaky_relu(self.deconv1_1_bn(self.deconv1_1(input)), 0.2)
        y = F.leaky_relu(self.deconv1_2_bn(self.deconv1_2(label)), 0.2)
        x = torch.cat([x, y], 1)
        x = F.leaky_relu(self.deconv2_bn(self.deconv2(x)), 0.2)
        x = F.leaky_relu(self.deconv3_bn(self.deconv3(x)), 0.2)
        x = F.leaky_relu(self.deconv4_bn(self.deconv4(x)), 0.2) 
        
        # added layer for 128x128
        x = F.leaky_relu(self.deconv5_bn(self.deconv5(x)), 0.2)

        # added layer for 256x256
        x = F.leaky_relu(self.deconv6_bn(self.deconv6(x)), 0.2) 

        # Final layer
        x = torch.tanh(self.final(x))
               
        return x

class Discriminator(nn.Module):
    def __init__(self, classes, d=64):
        super(Discriminator, self).__init__()
        self.conv1_1 = nn.Conv2d(3, int(d/2), 4, 2, 1)
        self.conv1_2 = nn.Conv2d(classes, int(d/2), 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d*8)
        
        # Additional layers to get to 128x128
        self.conv5 = nn.Conv2d(d*8, d*16, 4, 2, 1)
        self.conv5_bn = nn.BatchNorm2d(d*16)

        # Additional layers to get to 256x256
        self.conv6 = nn.Conv2d(d*16, d*32, 4, 2, 1)
        self.conv6_bn = nn.BatchNorm2d(d*32)

        # Final layer for 256x256
        self.final = nn.Conv2d(d*32, 1, 4, 1, 0)




    def forward(self, input, label):

        x = F.leaky_relu(self.conv1_1(input), 0.2) # 256x256x3 => 128x128x64
        y = F.leaky_relu(self.conv1_2(label), 0.2) # 256x256x10 => 128x128x64
        x = torch.cat([x, y], 1) # 128x128x128
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2) # 64x64x256
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2) # 32x32x512
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2) # 16x16x1024

        # added layer for 128x128
        x = F.leaky_relu(self.conv5_bn(self.conv5(x)), 0.2) # 16x16x1024

        # added layer for 256x256
        x = F.leaky_relu(self.conv6_bn(self.conv6(x)), 0.2) # 16x16x1024

        # Final layer
        x = torch.sigmoid(self.final(x)) # 1x1x1
        
        return x
