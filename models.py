import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets,models,transforms

class target_net():
    def __init__(self, model_name, num_classes, feature_extract, use_pretrained):
        super(target_net, self).__init__()  # MNIST:1*28*28  cifar:3*32*32
        self.model_name = model_name
        self.num_classes = num_classes
        self.feature_extract = feature_extract
        self.use_pretrained = use_pretrained


    def set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False  # 提取特征或变动所有权重

    def initialize_model(self):
        model_ft = None
        input_size = 0

        if self.model_name == 'resnet':
            model_ft = models.resnet18(pretrained=self.use_pretrained)
            self.set_parameter_requires_grad(model_ft, self.feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, self.num_classes)
            input_size = 224

        elif self.model_name == 'alexnet':
            model_ft = models.alexnet(pretrained=self.use_pretrained)
            self.set_parameter_requires_grad(model_ft, self.feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, self.num_classes)
            input_size = 224

        elif self.model_name == 'vgg':
            model_ft = models.vgg11_bn(pretrained=self.use_pretrained)
            self.set_parameter_requires_grad(model_ft, self.feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, self.num_classes)
            input_size = 224

        elif self.model_name == 'squeezenet':
            model_ft = models.squeezenet1_0(pretrained=self.use_pretrained)
            self.set_parameter_requires_grad(model_ft, self.feature_extract)
            model_ft.classifier[1] = nn.Conv2d(512, self.num_classes, kernel_size=(1, 1), stride=(1, 1))
            model_ft.num_classes = self.num_classes
            input_size = 224

        elif self.model_name == 'densenet':
            model_ft = models.densenet121(pretrained=self.use_pretrained)
            self.set_parameter_requires_grad(model_ft, self.feature_extract)
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = nn.Linear(num_ftrs, self.num_classes)
            input_size = 224

        elif self.model_name == 'inception':
            model_ft = models.inception_v3(pretrained=self.use_pretrained)
            self.set_parameter_requires_grad(model_ft, self.feature_extract)
            num_ftrs = model_ft.Auxlogits.fc.in_features
            model_ft.AuxLogits.fc = nn.Linear(num_ftrs, self.num_classes)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, self.num_classes)
            input_size = 299

        else:
            print('invalid model name,exiting...')
            exit()

        return model_ft, input_size



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # MNIST:1*28*28  cifar:3*32*32
        model = [
            nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=0, bias=True),
            nn.LeakyReLU(0.2),
            # 8*13*13  cifar:8*15*15
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            # 16*5*5   cifar:16*6*6
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 2, 2),
            nn.Sigmoid()
            # 32*1*1  cifar:32*1*1
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        output = self.model(x).squeeze()
        return output


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        encoder_lis = [
            # MNIST:1*28*28  cifar:3*32*32
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(8),
            nn.ReLU(),
            # 8*26*26   cifar:8*30*30
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            # 16*12*12   cifar:16*14*14
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            # 32*5*5   cifar:32*6*6
        ]

        bottle_neck_lis = [ResnetBlock(32),
                       ResnetBlock(32),
                       ResnetBlock(32),
                       ResnetBlock(32),]

        decoder_lis = [
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            # state size. 16 x 11 x 11   cifar:16*13*13
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(8),
            nn.ReLU(),
            # state size. 8 x 23 x 23    cifar:8*27*27
            nn.ConvTranspose2d(8, 3, kernel_size=6, stride=1, padding=0, bias=False),
            nn.Tanh()
            # state size. 3 x 28 x 28    cifar:3*32*32
        ]

        self.encoder = nn.Sequential(*encoder_lis)
        self.bottle_neck = nn.Sequential(*bottle_neck_lis)
        self.decoder = nn.Sequential(*decoder_lis)

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottle_neck(x)
        x = self.decoder(x)
        return x


# Define a resnet block
# modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out