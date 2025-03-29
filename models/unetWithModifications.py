import torch
import torch.nn as nn
import numpy as np
from monai.networks.blocks.dynunet_block import UnetResBlock

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class SpatialAdapter(torch.nn.Module):
    def __init__(self, adapter_kernel_size=3, feature_size=48, spatial_dims=3, norm_name="instance"):
        super().__init__()
        self.feature_size = feature_size
        self.adapter = UnetResBlock(spatial_dims, feature_size, feature_size, kernel_size=adapter_kernel_size,
                                    stride=1, norm_name=norm_name)

    def forward(self, x):
        out = self.adapter(x)

        return out



class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Creates a Double Convolutional Block with a series of two convolutional layers and ReLU activation.
        
        Arguments:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        """
        
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Applies the Double Convolutional Block to the input tensor.
        
        Arguments:
        x (Tensor): Input tensor.
        
        Returns:
        output (Tensor): Output tensor after applying the Double Convolutional Block.
        """
        
        return self.conv_op(x)

# changing init function so that I can access out_channels
# by extending UNet class
class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Creates a Downsampling Block with a series of convolutional layer and max pooling.
        
        Arguments:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        """
        
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.out_channels = out_channels

    def forward(self, x):
        """
        Applies the Downsampling Block to the input tensor.
        
        Arguments:
        x (Tensor): Input tensor.
        
        Returns:
        down (Tensor): Output tensor after downsampling.
        p (Tensor): Output tensor after max pooling.
        """
        
        down = self.conv(x)
        p = self.pool(down)

        return down, p


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Creates an Upsampling Block with a series of convolutional layers and upsampling.
        
        Arguments:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        """
        
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        Applies the Upsampling Block to the input tensors.
        
        Arguments:
        x1 (Tensor): Input tensor after downsampling.
        x2 (Tensor): Input tensor before downsampling.
        
        Returns:
        x (Tensor): Output tensor after upsampling and concatenating.
        """
        
        x1 = self.up(x1)
        x = torch.cat([x1, x2], 1)
        
        return self.conv(x)
   

class SharedDown(nn.Module):
  def __init__(self,in_channels, random_seed):
    """
    Creates a seperate model for all the
    down blocks of Unet
    """
    super().__init__()
    self.down_convolution_1 = DownSample(in_channels, 64)
    self.down_convolution_2 = DownSample(64, 128)
    self.down_convolution_3 = DownSample(128, 256)
    self.down_convolution_4 = DownSample(256, 512)

    self.bottle_neck = DoubleConv(512, 1024)
    self.output_channels = 1024

    generator = torch.Generator().manual_seed(random_seed)
    self.initialize_weights(generator)

  def forward(self,x):

    down_1, p1 = self.down_convolution_1(x)
    down_2, p2 = self.down_convolution_2(p1)
    down_3, p3 = self.down_convolution_3(p2)
    down_4, p4 = self.down_convolution_4(p3)

    b = self.bottle_neck(p4) 
    
    return (
      b,
      down_4,
      down_3,
      down_2,
      down_1
    ) 
  
  def initialize_weights(self, generator):
    """
    Initializes the weights of the model using Kaiming Normal initialization.

    Arguments:
    generator (torch.Generator): Random number generator for replication.
    """

    for module in self.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu', generator=generator)
            
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

class QGenerator(nn.Module):
  def __init__(self, in_channels, random_seed, kernel_size=1):
    super().__init__()
    self.in_channels = in_channels
    # Using in_channels/2 balances expressiveness 
    # and computational efficiency.
    self.attn_dim = in_channels
    self.kernel_size = kernel_size
    self.query = nn.Conv2d(self.in_channels, self.attn_dim, kernel_size=self.kernel_size)

    generator = torch.Generator().manual_seed(random_seed)
    self.initialize_weights(generator)

  def forward(self,b):
    pass

    B, C, H, W = b.shape  # Batch, Channels, Height, Width

    # Generate only Q
    # Q will have shape (B, H*W, attn_dim)
    Q = self.query(b).view(B, -1, H * W).permute(0, 2, 1)

    return Q

  def initialize_weights(self, generator):
    """
    Initializes the weights of the model using Kaiming Normal initialization.

    Arguments:
    generator (torch.Generator): Random number generator for replication.
    """

    for module in self.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu', generator=generator)
            
            if module.bias is not None:
                nn.init.constant_(module.bias, 0) 

class SharedUpWithAttn(nn.Module):
  def __init__(self, in_channels, num_classes, random_seed):
    super().__init__()
    self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1)
    self.key = nn.Conv2d(in_channels, in_channels, kernel_size=1)
    self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
    
    # Softmax for attention scores
    self.softmax = nn.Softmax(dim=-1)
    # self.conv_expand = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
    # self.bn_attn = nn.InstanceNorm2d(in_channels, affine=True)
    
    self.up_convolution_1 = UpSample(in_channels, 512)
    self.up_convolution_2 = UpSample(512, 256)
    self.up_convolution_3 = UpSample(256, 128)
    self.up_convolution_4 = UpSample(128, 64)

    self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    generator = torch.Generator().manual_seed(random_seed)
    self.initialize_weights(generator)

  def forward(self,Q,feature_tuple):
    b,down_4,down_3,down_2,down_1 = feature_tuple

    B, C, H, W = b.shape
    # Q1 = self.query(b).view(B,-1,H*W)
    K = self.key(b).view(B, -1, H * W)                    
    V = self.value(b).view(B, -1, H * W).permute(0, 2, 1)

    attn_scores = torch.bmm(Q, K) / (K.shape[1] ** 0.5)
    attn_map = self.softmax(attn_scores)

    attn_out = torch.bmm(attn_map, V)  # (B, H*W, attn_dim)
    attn_out = attn_out.permute(0, 2, 1).contiguous().view(B, -1, H, W)
    # up_input = torch.cat([attn_out, self.conv_expand(attn_out)], dim=1)
    # up_input = self.bn_attn(up_input)
    print(f'attn_out dtype : {attn_out.dtype}')
    # exit()
    up_input = attn_out
    # up_input = self.bn_attn(up_input)

    up_1 = self.up_convolution_1(up_input, down_4)
    up_2 = self.up_convolution_2(up_1, down_3)
    up_3 = self.up_convolution_3(up_2, down_2)
    up_4 = self.up_convolution_4(up_3, down_1)
    
    return self.out(up_4)
        
  
  def initialize_weights(self, generator):
    """
    Initializes the weights of the model using Kaiming Normal initialization.

    Arguments:
    generator (torch.Generator): Random number generator for replication.
    """

    for module in self.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu', generator=generator)
            
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)





class SharedUnet(nn.Module):
  def __init__(self, in_channels, num_classes, random_seed):
    """
    Creates a UNetWithAdapter model for semantic segmentation.

    Arguments:
    in_channels (int): Number of input channels.
    num_classes (int): Number of classes in the dataset.
    """

    super().__init__()
    self.down_convolution_1 = DownSample(in_channels, 64)
    self.down_convolution_2 = DownSample(64, 128)
    self.down_convolution_3 = DownSample(128, 256)
    self.down_convolution_4 = DownSample(256, 512)

    self.bottle_neck = DoubleConv(512, 1024)

    self.up_convolution_1 = UpSample(1024, 512)
    self.up_convolution_2 = UpSample(512, 256)
    self.up_convolution_3 = UpSample(256, 128)
    self.up_convolution_4 = UpSample(128, 64)

    self.featureSizeForAdapter = 64

    # Apply weight initialization
    generator = torch.Generator().manual_seed(random_seed)
    self.initialize_weights(generator)

  def forward(self, x):
    """
    Applies the UNet model to the input tensor.

    Arguments:
    x (Tensor): Input tensor.

    Returns:
    output (Tensor): Output tensor after applying the UNet model.
    """

    down_1, p1 = self.down_convolution_1(x)
    down_2, p2 = self.down_convolution_2(p1)
    down_3, p3 = self.down_convolution_3(p2)
    down_4, p4 = self.down_convolution_4(p3)

    b = self.bottle_neck(p4)

    up_1 = self.up_convolution_1(b, down_4)
    up_2 = self.up_convolution_2(up_1, down_3)
    up_3 = self.up_convolution_3(up_2, down_2)
    up_4 = self.up_convolution_4(up_3, down_1)

    # adapted = self.adapter(up_4)

    # output = self.out(adapted)

    return up_4

  def initialize_weights(self, generator):
    """
    Initializes the weights of the model using Kaiming Normal initialization.

    Arguments:
    generator (torch.Generator): Random number generator for replication.
    """

    for module in self.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu', generator=generator)
            
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)



class PersonalizedUnet(nn.Module):
  def __init__(self, in_channels, num_classes, random_seed):
    """
    Creates a UNetWithAdapter model for semantic segmentation.

    Arguments:
    in_channels (int): Number of input channels.
    num_classes (int): Number of classes in the dataset.
    """

    super().__init__()

    self.adapter = SpatialAdapter(feature_size=in_channels,spatial_dims=2)

    self.out = nn.Conv2d(in_channels=in_channels, out_channels=num_classes, kernel_size=1)

    # Apply weight initialization
    generator = torch.Generator().manual_seed(random_seed)
    self.initialize_weights(generator)

  def forward(self, x):
    """
    Applies the UNet model to the input tensor.

    Arguments:
    x (Tensor): Input tensor.

    Returns:
    output (Tensor): Output tensor after applying the UNet model.
    """

    adapted = self.adapter(x)

    output = self.out(adapted)

    return output

  def initialize_weights(self, generator):
    """
    Initializes the weights of the model using Kaiming Normal initialization.

    Arguments:
    generator (torch.Generator): Random number generator for replication.
    """

    for module in self.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu', generator=generator)
            
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)