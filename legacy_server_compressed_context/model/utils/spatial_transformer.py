import torch
from torch import nn
from torch.nn import functional as F
from timm.layers import LayerNorm
from einops import rearrange, repeat
import numpy as np 
from tools.utils import save_image, normalize

class Spatial_transformers(nn.Module):
    def __init__(self, SH=224, intermediate=250, input_channel=3, kernel_size=5, padding=2, 
        output_spatial_dim=None):
        super().__init__()
        
        self.SH = SH
        
        if output_spatial_dim is None:
            output_shape = SH // 2 // 2 // 2
            output_dim = output_shape * output_shape * intermediate
            self.st = nn.Sequential(
                nn.MaxPool2d(2, stride=2, ceil_mode=True),
                nn.Conv2d(input_channel, intermediate, kernel_size=kernel_size ,stride=1, padding=padding),
                nn.ReLU(True),
                nn.MaxPool2d(2, stride=2 , ceil_mode=True),
                nn.Conv2d(intermediate, intermediate, kernel_size=kernel_size ,stride=1, padding=padding),
                nn.ReLU(True),
                nn.MaxPool2d(2, stride=2 , ceil_mode=True)
            )
            self.st[1].weight.data.zero_()
            self.st[1].bias.data.zero_()
            self.st[4].weight.data.zero_()
            self.st[4].bias.data.zero_()
            
        else:
            output_shape = SH
            output_dim = output_spatial_dim
        
            self.st = nn.Sequential(
                nn.Conv2d(input_channel, intermediate, kernel_size=kernel_size ,stride=1, padding=padding),
                LayerNorm([intermediate, SH, SH]),
                nn.ReLU(True),
                nn.Conv2d(intermediate, intermediate, kernel_size=kernel_size ,stride=1, padding=padding),
                LayerNorm([intermediate, SH, SH]),
            )
            self.st[0].weight.data.zero_()
            self.st[0].bias.data.zero_()
            self.st[3].weight.data.zero_()
            self.st[3].bias.data.zero_()
            
    
        self.FC_ = nn.Sequential(
            nn.Linear(output_dim, intermediate),
            nn.ReLU(True),
            nn.Linear( intermediate , 6 )
        )

        self.FC_[0].weight.data.zero_()
        self.FC_[0].bias.data.zero_()

        self.FC_[2].weight.data.zero_()
        self.FC_[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x, BNC=None):
        if BNC:
            cl_token = x[:,0]
            N = x.shape[1] -1
            x = rearrange(x[:,1:], "B (H W) C -> B C H W", H = int(N **0.5), W = int(N **0.5))
            # y = self.st[0](x)
            # y = self.st[1](y)

        H = self.st(x)
        # torch.Size([B, C=250, 28, 28])
        H = rearrange(H, "B C H W -> B (C H W)")
        H = self.FC_(H)

        theta = H.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())

        x = F.grid_sample(x, grid)

        if BNC:
            x = rearrange(x, "B C H W -> B (H W) C ")
            x = torch.cat(( cl_token.unsqueeze(1), x), dim=1)
        
        return x

class Spatial_transformers2(nn.Module):
    def __init__(self, input_channel=3, Height=224, Width=224, drop_prob=0.5, intermediate=20):
        super(Spatial_transformers2, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=input_channel * Height * Width, out_features=intermediate),
            # LayerNorm(intermediate),
            nn.Tanh(),
            nn.Dropout(drop_prob),
            nn.Linear(in_features=intermediate, out_features=6),
            nn.Tanh(),
        )
        bias = torch.from_numpy(np.array([1, 0, 0, 0, 1, 0]))
        # nn.init.constant_(self.fc[3].weight, 0)
        self.fc[3].bias.data.copy_(bias)
    
    def forward(self, img, BNC=None):
        '''
        :param img: (b, c, h, w)
        :return: (b, c, h, w)
        '''
        if BNC:
            cl_token = img[:,0]
            N = img.shape[1] -1
            img = rearrange(img[:,1:], "B (H W) C -> B C H W", H = int(N **0.5), W = int(N **0.5))
            # y = self.st[0](x)
            # y = self.st[1](y)


        batch_size = img.size(0)
        theta = self.fc(img.reshape(batch_size, -1)).view(batch_size, 2, 3)
        grid = F.affine_grid(theta,  img.size() )
        img_transform = F.grid_sample(img, grid)

        if BNC:
            img_transform = rearrange(img_transform, "B C H W -> B (H W) C ")
            img_transform = torch.cat(( cl_token.unsqueeze(1), img_transform), dim=1)
        
        return img_transform


class Spatial_transformers3(nn.Module):
    def __init__(self, input_channel=3, Height=224, Width=224, drop_prob=0.5, intermediate=20):
        super(Spatial_transformers3, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=input_channel * Height * Width, out_features=intermediate),
            # LayerNorm(intermediate),
            nn.Tanh(),
            nn.Dropout(drop_prob),
            nn.Linear(in_features=intermediate, out_features=2),
            nn.Tanh(),
        )
            
    def forward(self, img, BNC=None):
        if BNC:
            cl_token = img[:,0]
            N = img.shape[1] -1
            img = rearrange(img[:,1:], "B (H W) C -> B C H W", H = int(N **0.5), W = int(N **0.5))
            # y = self.st[0](x)
            # y = self.st[1](y)

        batch_size = img.size(0)
        params = self.fc(img.reshape(batch_size, -1))
        
        # save_image(normalize( img ), "temp.png")    
        theta = torch.zeros((batch_size, 2, 3)).cuda()
        theta[:,0,0 ] = theta[:,1,1 ] = params[:,0] + 1
        theta[:,0,1 ] = params[:,1]
        theta[:,1,0 ] = -theta[:,0,1 ]

        grid = F.affine_grid(theta,  img.size() )
        img_transform = F.grid_sample(img, grid)
        
        # save_image(normalize( img_transform ), "temp2.png")

        if BNC:
            img_transform = rearrange(img_transform, "B C H W -> B (H W) C ")
            img_transform = torch.cat(( cl_token.unsqueeze(1), img_transform), dim=1)
        
        return img_transform

