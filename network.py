import torch.nn as nn

## ResnetBlock
class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim)

    def build_conv_block(self, dim):
        conv_block = []

        conv_block += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True)
        ]

        return nn.Sequential(*conv_block)

    def forward(self, input):
        output = input + self.conv_block(input)
        return output

# Generator
class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_block=7):
        super(Generator, self).__init__()

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=True),
                 nn.InstanceNorm2d(ngf),
                 nn.ReLU(True)]

        ## downsampling
        n_downsampling = 3
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=True),
                      nn.InstanceNorm2d(ngf * mult * 2),
                      nn.ReLU(True)]

        ## resblock
        mult = 2 ** n_downsampling
        for i in range(n_block):
            model += [ResnetBlock(ngf * mult)]

        ## upsampling
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
                      nn.InstanceNorm2d(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_nc, output_nc, ndf):
        super(Discriminator, self).__init__()

        model = [nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1, bias=True),
                 nn.LeakyReLU(True)]

        for i in range(3):
            mult = 2 ** i
            model += [nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=2, padding=1, bias=True),
                      nn.InstanceNorm2d(ndf * mult * 2),
                      nn.LeakyReLU(True)]

        model += [nn.Conv2d(ndf * mult * 2, 1, kernel_size=4, stride=1, padding=0, bias=True),
                  nn.Sigmoid()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        out = self.model(input)
        return out