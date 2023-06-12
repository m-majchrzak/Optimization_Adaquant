import torch
import torch.nn as nn
import math
from utils.quantize import QConv2d, QLinear

class Lambda(nn.Module):
    def __init__(self):
        super(Lambda, self).__init__()

    def forward(self,x):
        return x

def depBatchNorm2d(exists, *kargs, **kwargs):
    if exists:
        return nn.BatchNorm2d(*kargs, **kwargs)
    else:
        return Lambda()


def conv3x3(in_planes, out_planes, stride=1, groups=1, bias=False,num_bits=8,num_bits_weight=8,measure=False, cal_qparams=False):
    "3x3 convolution with padding"
    return QConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=bias,num_bits=num_bits,num_bits_weight=num_bits_weight,measure=measure, cal_qparams=cal_qparams)


def init_model(model):
    for m in model.modules():
        if isinstance(m, QConv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    for m in model.modules():
        if isinstance(m, Bottleneck):
            nn.init.constant_(m.bn3.weight, 0)
        elif isinstance(m, BasicBlock):
            nn.init.constant_(m.bn2.weight, 0)

    model.fc.weight.data.normal_(0, 0.01)
    model.fc.bias.data.zero_()


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes,  stride=1, expansion=1, downsample=None, groups=1, residual_block=None,batch_norm=True,measure=False,num_bits=8,num_bits_weight=8, cal_qparams=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, groups=groups,bias=not batch_norm,num_bits=num_bits,num_bits_weight=num_bits_weight,measure=measure, cal_qparams=cal_qparams)
        self.bn1 = depBatchNorm2d(batch_norm,planes)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, expansion * planes, groups=groups,bias=not batch_norm,num_bits=num_bits,num_bits_weight=num_bits_weight,measure=measure, cal_qparams=cal_qparams)
        self.bn2 = depBatchNorm2d(batch_norm, expansion * planes)
        self.relu2 = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.residual_block = residual_block
        self.stride = stride
        self.expansion = expansion

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        #import pdb; pdb.set_trace()
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(residual)

        if self.residual_block is not None:
            residual = self.residual_block(residual)
        out += residual
        out = self.relu2(out)
        return out


class Bottleneck(nn.Module):

    def __init__(self, inplanes, planes,  stride=1, expansion=4, downsample=None, groups=1, residual_block=None,batch_norm=True,measure=False,num_bits=8,num_bits_weight=8, cal_qparams=False):
        super(Bottleneck, self).__init__()
        self.conv1 = QConv2d(
            inplanes, planes, kernel_size=1, bias=not batch_norm,num_bits=num_bits,num_bits_weight=num_bits_weight,measure=measure, cal_qparams=cal_qparams)
        self.bn1 = depBatchNorm2d(batch_norm, planes)
        self.conv2 = conv3x3(planes, planes, stride=stride, groups=groups,bias=not batch_norm,num_bits=num_bits,num_bits_weight=num_bits_weight,measure=measure, cal_qparams=cal_qparams)
        self.bn2 = depBatchNorm2d(batch_norm, planes)
        self.conv3 = QConv2d(
            planes, planes * expansion, kernel_size=1, bias=not batch_norm,num_bits=num_bits,num_bits_weight=num_bits_weight,measure=measure, cal_qparams=cal_qparams)
        self.bn3 = depBatchNorm2d(batch_norm, planes * expansion)
        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)
        self.relu3 = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.residual_block = residual_block
        self.stride = stride
        self.expansion = expansion

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        if self.residual_block is not None:
            residual = self.residual_block(residual)

        out += residual
        out = self.relu3(out)

        return out


class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()

    def _make_layer(self, block, planes, blocks, expansion=1, stride=1, groups=1, residual_block=None, batch_norm=True, num_bits=8, num_bits_weight=8, perC=True,measure=False, cal_qparams=False):
        downsample = None
        out_planes = planes * expansion
        if stride != 1 or self.inplanes != out_planes:
            downsample = nn.Sequential(
                QConv2d(self.inplanes, out_planes,
                          kernel_size=1, stride=stride, bias=not batch_norm,num_bits=num_bits,num_bits_weight=num_bits_weight,measure=measure, cal_qparams=cal_qparams),
                depBatchNorm2d(batch_norm,planes * expansion),
            )
        if residual_block is not None:
            residual_block = residual_block(out_planes)

        layers = []
        layers.append(block(self.inplanes, planes, stride, expansion=expansion,
                            downsample=downsample, groups=groups, residual_block=residual_block,batch_norm=batch_norm,num_bits=num_bits,num_bits_weight=num_bits_weight,measure=measure, cal_qparams=cal_qparams))
        self.inplanes = planes * expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, expansion=expansion, groups=groups,
                                residual_block=residual_block,batch_norm=batch_norm,num_bits=num_bits,num_bits_weight=num_bits_weight,measure=measure, cal_qparams=cal_qparams))

        return nn.Sequential(*layers)

    def features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        return x.view(x.size(0), -1)

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x

    @staticmethod
    def regularization_pre_step(model, weight_decay=1e-4):
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m, QConv2d) or isinstance(m, nn.Linear):
                    if m.weight.grad is not None:
                        m.weight.grad.add_(weight_decay * m.weight)
        return 0


class ResNet_imagenet(ResNet):

    def __init__(self, num_classes=1000, inplanes=64,
                 block=Bottleneck, residual_block=None, layers=[3, 4, 23, 3],
                 width=[64, 128, 256, 512], expansion=4, groups=[1, 1, 1, 1],
                 regime='normal', scale_lr=1,batch_norm=True,num_bits=8,num_bits_weight=8, perC=True, measure=False, cal_qparams=False):
        super(ResNet_imagenet, self).__init__()
        self.inplanes = inplanes
        self.conv1 = QConv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=not batch_norm,num_bits=num_bits,num_bits_weight=num_bits_weight, perC=perC ,measure=measure, cal_qparams=cal_qparams)
        self.bn1 = depBatchNorm2d(batch_norm,self.inplanes)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        for i in range(len(layers)):
            #if i==2 or i==1: 
            #    print(i)
            #    num_bits = 4
            #    num_bits_weight = 4
            setattr(self, 'layer%s' % str(i + 1),
                    self._make_layer(block=block, planes=width[i], blocks=layers[i], expansion=expansion,
                                     stride=1 if i == 0 else 2, residual_block=residual_block, groups=groups[i],batch_norm=batch_norm,num_bits=num_bits,num_bits_weight=num_bits_weight,perC=perC, measure=measure, cal_qparams=cal_qparams))

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = QLinear(width[-1] * expansion, num_classes,num_bits_weight=num_bits_weight,perC=perC,measure=measure, cal_qparams=cal_qparams)
        if batch_norm:
            init_model(self)

        def ramp_up_lr(lr0, lrT, T):
            rate = (lrT - lr0) / T
            return "lambda t: {'lr': %s + t * %s}" % (lr0, rate)
        if regime == 'normal':
            self.regime = [
                {'epoch': 0, 'optimizer': 'SGD', 'momentum': 0.9,
                    'step_lambda': ramp_up_lr(0.1, 0.1 * scale_lr, 5004 * 5 / scale_lr)},
                {'epoch': 5,  'lr': scale_lr * 1e-1},
                {'epoch': 30, 'lr': scale_lr * 1e-2},
                {'epoch': 60, 'lr': scale_lr * 1e-3},
                {'epoch': 80, 'lr': scale_lr * 1e-4}
            ]
        elif regime == 'fast':
            self.regime = [
                {'epoch': 0, 'optimizer': 'SGD', 'momentum': 0.9,
                    'step_lambda': ramp_up_lr(0.1, 0.1 * 4 * scale_lr, 5004 * 4 / (4 * scale_lr))},
                {'epoch': 4,  'lr': 4 * scale_lr * 1e-1},
                {'epoch': 18, 'lr': scale_lr * 1e-1},
                {'epoch': 21, 'lr': scale_lr * 1e-2},
                {'epoch': 35, 'lr': scale_lr * 1e-3},
                {'epoch': 43, 'lr': scale_lr * 1e-4},
            ]
            self.data_regime = [
                {'epoch': 0, 'input_size': 128, 'batch_size': 256},
                {'epoch': 18, 'input_size': 224, 'batch_size': 64},
                {'epoch': 41, 'input_size': 288, 'batch_size': 32},
            ]
        elif regime == 'small':
            scale_lr *= 4
            self.regime = [
                {'epoch': 0, 'optimizer': 'SGD',
                    'momentum': 0.9, 'lr': scale_lr * 1e-1},
                {'epoch': 30, 'lr': scale_lr * 1e-2},
                {'epoch': 60, 'lr': scale_lr * 1e-3},
                {'epoch': 80, 'lr': scale_lr * 1e-4}
            ]
            self.data_regime = [
                {'epoch': 0, 'input_size': 128, 'batch_size': 256},
                {'epoch': 80, 'input_size': 224, 'batch_size': 64},
            ]
            self.data_eval_regime = [
                {'epoch': 0, 'input_size': 128, 'batch_size': 1024},
                {'epoch': 80, 'input_size': 224, 'batch_size': 512},
            ]

    
