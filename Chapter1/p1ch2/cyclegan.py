import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

class ResNetBlock(nn.Module):
    """
    Residual block for the ResNetGenerator.
    """

    def __init__(self, dim):
        super(ResNetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim)

    def build_conv_block(self, dim):
        """
        Build the convolutional block for the ResNetBlock.

        Args:
            dim (int): Number of input and output channels.

        Returns:
            nn.Sequential: Convolutional block.
        """
        conv_block = []

        conv_block += [nn.ReflectionPad2d(1)]

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
                       nn.InstanceNorm2d(dim),
                       nn.ReLU(True)]

        conv_block += [nn.ReflectionPad2d(1)]

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
                       nn.InstanceNorm2d(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """
        Forward pass of the ResNetBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        out = x + self.conv_block(x)
        return out


class ResNetGenerator(nn.Module):
    """
    Residual Network (ResNet) generator for image-to-image translation.
    """

    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=9):
        """
        Initialize the ResNetGenerator.

        Args:
            input_nc (int): Number of input channels.
            output_nc (int): Number of output channels.
            ngf (int): Number of filters in the generator.
            n_blocks (int): Number of ResNet blocks.
        """
        assert(n_blocks >= 0)
        super(ResNetGenerator, self).__init__()

        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=True),
                 nn.InstanceNorm2d(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=True),
                      nn.InstanceNorm2d(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResNetBlock(ngf * mult)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=True),
                      nn.InstanceNorm2d(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """
        Forward pass of the ResNetGenerator.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.model(input)

netG = ResNetGenerator()
model_path = '../Data/horse2zebra_0.4.0.pth'
model_data = torch.load(model_path)
netG.load_state_dict(model_data)

print(netG.eval())

preprocess = transforms.Compose([transforms.Resize(256),
                                 transforms.ToTensor()])

img = Image.open(r'C:\Users\brand\Documents\Data Science Statistics Book\Pytorch_learning\PyTorchPractice\Chapter1\Data\bobby.jpg')
img_t = preprocess(img)
batch_t = torch.unsqueeze(img_t, 0)

batch_out = netG(batch_t)

out_t = (batch_out.data.squeeze() + 1.0) / 2.0
out_img = transforms.ToPILImage()(out_t)
out_img.save('../Data/zebra.jpg')
out_img
