
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class ZNet_v2(nn.Module):
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal(m.weight.data)

    def __init__(self):
        super(ZNet_v2, self).__init__()
        self.down1 = conv_block(in_channels=3,out_channels=64,kernel_size=(7, 7),stride=(2, 2),padding=(3, 3),
        )
        self.down2a = mod_Inception_block(64)
        # self.down2b = Inception_block(128)
        # self.down2b = Inception_block(in_channels=64, out_1x1=10, red_3x3=16, out_3x3=24, red_5x5=10, out_5x5=20, out_1x1pool=10)
        # self.down2c = Inception_block(256)

        self.down3a = mod_Inception_block(232)
        # self.down3d = Inception_block(in_channels=128, out_1x1=82, red_3x3=36, out_3x3=114, red_5x5=15, out_5x5=60)
        
        
        self.down4a = mod_Inception_block(736)
        # self.down4e = Inception_block(in_channels=256, out_1x1=242, red_3x3=40, out_3x3=140, red_5x5=36, out_5x5=130)

        # self.neck_a = Inception_block(in_channels=512, out_1x1=242, red_3x3=40, out_3x3=140, red_5x5=36, out_5x5=130)

        self.up1 = upsample_block(in_channels=2248, out_1x1=82, red_3x3=36, out_3x3=114, red_5x5=15, out_5x5=60)
        self.up2 = upsample_block(in_channels=256, out_1x1=40, red_3x3=16, out_3x3=48, red_5x5=10, out_5x5=40)
        self.up3 = upsample_block(in_channels=128, out_1x1=20, red_3x3=12, out_3x3=28, red_5x5=4, out_5x5=16)
        self.output = upsample_block(in_channels=64, out_1x1=2, red_3x3=8, out_3x3=3, red_5x5=8, out_5x5=3)

        self.apply(self.weight_init)

        
    def forward(self, m):
                            #[3,3,256,256] input
        x0 = self.down1(m)   # [3,64,128,128]
        

        m = self.down2a(x0)  # [3,64,64,64]
        # m = self.down2b(m)
        # x1 = self.down2c(m)  # [3,128,64,64]

       
        m = self.down3a(m)  # [3,128,32,32]
        # x2 = self.down3d(m)  # [3,256,32,32]  

        m = self.down4a(m) #  [3,256,16,16]
        # m = self.down4e(m)  #  [3,512,16,16]


        # m = self.neck_a(m)  

        m = self.up1(m)   # [3,256,32,32]
        # m = torch.cat([m,x2], dim=1)
        m = self.up2(m)   # [3,128,64,64]
        # m = torch.cat([m,x1], dim=1)
        m = self.up3(m)   # [3,64,128,128]
        # m = torch.cat([m,x0], dim=1)
        m = self.output(m)  #[3,6,256,256]
        # m = F.log_softmax(m)
        return m


class mod_Inception_block(nn.Module):
    def __init__(
        self, in_channels):
        super(mod_Inception_block, self).__init__()
        self.branch1 =nn.Sequential( Dense_Block7(4,in_channels,2,1),
        nn.MaxPool2d(kernel_size=(7, 7), stride=(2, 2), padding=3))

        self.branch2 = nn.Sequential( Dense_Block(8,in_channels,2,1),
        nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1))

        self.branch3 = nn.Sequential( Dense_Block5(8,in_channels,2,1),
        nn.MaxPool2d(kernel_size=(5, 5), stride=(2, 2), padding=2))                                                     

    def forward(self, x):
        return torch.cat(
            [self.branch1(x), self.branch2(x), self.branch3(x)], 1
            ) 



# class Inception_block(nn.Module):
#     def __init__(
#         self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5
#     ):
#         super(Inception_block, self).__init__()
#         self.branch1 = conv_block(in_channels, out_1x1, kernel_size=(1, 1))

#         self.branch2 = nn.Sequential(
#             conv_block(in_channels, red_3x3, kernel_size=(1, 1)),
#             conv_block(red_3x3, out_3x3, kernel_size=(3, 3), padding=(1, 1)),              # Recieve 28x28 output
#         )

#         self.branch3 = nn.Sequential(
#             conv_block(in_channels, red_5x5, kernel_size=(1, 1)),
#             conv_block(red_5x5, out_5x5, kernel_size=(5, 5), padding=(2, 2)),
#         )                                                                    

#     def forward(self, x):
#         return torch.cat(
#             [self.branch1(x), self.branch2(x), self.branch3(x)], 1
#         )

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(conv_block, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))



class upsample_block(nn.Module):
    def __init__(
        self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5):
        super(upsample_block, self).__init__()
        self.branch1 = nn.Sequential(
            conv_block(in_channels, out_1x1, kernel_size=(1, 1)),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  
        )

        self.branch2 = nn.Sequential(
            conv_block(in_channels, red_3x3, kernel_size=(1, 1)), 
            conv_block(red_3x3, out_3x3, kernel_size=(3, 3), padding=(1, 1)),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)            
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, red_5x5, kernel_size=(1, 1)),
            conv_block(red_5x5, out_5x5, kernel_size=(5, 5), padding=(2, 2)),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
                                                                    
    

    def forward(self, x):
        return torch.cat(
            [self.branch1(x), self.branch2(x), self.branch3(x)], 1)



class Dense_Layer(nn.Module):
    def __init__(self, in_channels, growthrate, bn_size):
        super(Dense_Layer, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels, bn_size * growthrate, kernel_size=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(bn_size * growthrate)
        self.conv2 = nn.Conv2d(
            bn_size * growthrate, growthrate, kernel_size=3, padding=1, bias=False
        )

    def forward(self, prev_features):
        out1 = torch.cat(prev_features, dim=1)
        out1 = self.conv1(F.relu(self.bn1(out1)))
        out2 = self.conv2(F.relu(self.bn2(out1)))
        return out2

class Dense_Layer5(nn.Module):
    def __init__(self, in_channels, growthrate, bn_size):
        super(Dense_Layer5, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels, bn_size * growthrate, kernel_size=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(bn_size * growthrate)
        self.conv2 = nn.Conv2d(
            bn_size * growthrate, growthrate, kernel_size=5, padding=2, bias=False
        )

    def forward(self, prev_features):
        out1 = torch.cat(prev_features, dim=1)
        out1 = self.conv1(F.relu(self.bn1(out1)))
        out2 = self.conv2(F.relu(self.bn2(out1)))
        return out2

class Dense_Block(nn.ModuleDict):
    def __init__(self, n_layers, in_channels, growthrate, bn_size):
        """
        A Dense block consists of `n_layers` of `Dense_Layer`
        Parameters
        ----------
            n_layers: Number of dense layers to be stacked 
            in_channels: Number of input channels for first layer in the block
            growthrate: Growth rate (k) as mentioned in DenseNet paper
            bn_size: Multiplicative factor for # of bottleneck layers
        """
        super(Dense_Block, self).__init__()

        layers = dict()
        for i in range(n_layers):
            layer = Dense_Layer(in_channels + i * growthrate, growthrate, bn_size)
            layers['dense{}'.format(i)] = layer
        
        self.block = nn.ModuleDict(layers)
    
    def forward(self, features):
        if(isinstance(features, torch.Tensor)):
            features = [features]
        
        for _, layer in self.block.items():
            new_features = layer(features)
            features.append(new_features)

        return torch.cat(features, dim=1)

class Dense_Block5(nn.ModuleDict):
    def __init__(self, n_layers, in_channels, growthrate, bn_size):
        """
        A Dense block consists of `n_layers` of `Dense_Layer`
        Parameters
        ----------
            n_layers: Number of dense layers to be stacked 
            in_channels: Number of input channels for first layer in the block
            growthrate: Growth rate (k) as mentioned in DenseNet paper
            bn_size: Multiplicative factor for # of bottleneck layers
        """
        super(Dense_Block5, self).__init__()

        layers = dict()
        for i in range(n_layers):
            layer = Dense_Layer5(in_channels + i * growthrate, growthrate, bn_size)
            layers['dense{}'.format(i)] = layer
        
        self.block = nn.ModuleDict(layers)
    
    def forward(self, features):
        if(isinstance(features, torch.Tensor)):
            features = [features]
        
        for _, layer in self.block.items():
            new_features = layer(features)
            features.append(new_features)

        return torch.cat(features, dim=1)
    
class Dense_Block7(nn.ModuleDict):
    def __init__(self, n_layers, in_channels, growthrate, bn_size):
        """
        A Dense block consists of `n_layers` of `Dense_Layer`
        Parameters
        ----------
            n_layers: Number of dense layers to be stacked 
            in_channels: Number of input channels for first layer in the block
            growthrate: Growth rate (k) as mentioned in DenseNet paper
            bn_size: Multiplicative factor for # of bottleneck layers
        """
        super(Dense_Block7, self).__init__()

        layers = dict()
        for i in range(n_layers):
            layer = Dense_Layer7(in_channels + i * growthrate, growthrate, bn_size)
            layers['dense{}'.format(i)] = layer
        
        self.block = nn.ModuleDict(layers)
    
    def forward(self, features):
        if(isinstance(features, torch.Tensor)):
            features = [features]
        
        for _, layer in self.block.items():
            new_features = layer(features)
            features.append(new_features)

        return torch.cat(features, dim=1)

class Dense_Layer7(nn.Module):
    def __init__(self, in_channels, growthrate, bn_size):
        super(Dense_Layer7, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels, bn_size * growthrate, kernel_size=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(bn_size * growthrate)
        self.conv2 = nn.Conv2d(
            bn_size * growthrate, growthrate, kernel_size=7, padding=3, bias=False
        )

    def forward(self, prev_features):
        out1 = torch.cat(prev_features, dim=1)
        out1 = self.conv1(F.relu(self.bn1(out1)))
        out2 = self.conv2(F.relu(self.bn2(out1)))
        return out2
              
if __name__ == "__main__":
    from torchscope import scope
    model = ZNet_v2()
    scope(model, input_size=(3, 256, 256))
