import torch
from torch import nn
import torch.nn.functional as F

import model.resnet as models
from model.memory import Memory
import model.mobilenet as model_MV2

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP(nn.Module):
    def __init__(self, backbone, output_stride):
        super(ASPP, self).__init__()
        if backbone == 'drn':
            inplanes = 512
        elif backbone == 'mobilenet':
            inplanes = 320
        else:
            inplanes = 2048
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0])
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3])

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(256),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



class Deeplabv3_mem(nn.Module):
    def __init__(self, layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=2, zoom_factor=8, use_ppm=True,
        backbone = 'resnet', output_stride=8, memory_size=20, criterion=nn.CrossEntropyLoss(ignore_index=255),
                 pretrained=True):
        super(Deeplabv3_mem, self).__init__()
        assert layers in [18, 34, 50, 101, 152]
        assert 2048 % len(bins) == 0
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.use_ppm = use_ppm
        self.criterion = criterion
        self.memory_size = memory_size
        self.backbone=backbone

        if backbone == 'mobilenet':
            self.mobilenet = model_MV2.MobileNetV2(output_stride=output_stride, pretrained=pretrained)
        else:
            if layers == 18:
                resnet = models.resnet18(pretrained=pretrained, deep_base=False)
            elif layers == 34:
                resnet = models.resnet34(pretrained=pretrained, deep_base=False)
            elif layers == 50:
                resnet = models.resnet50(pretrained=pretrained)
            elif layers == 101:
                resnet = models.resnet101(pretrained=pretrained)
            else:
                resnet = models.resnet152(pretrained=pretrained)

            if layers >= 50:
                self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2,
                                            resnet.relu, resnet.conv3, resnet.bn3, resnet.relu, resnet.maxpool)
            else:
                self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
            self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        if backbone == 'drn':
            inplanes = 512
        elif backbone == 'mobilenet':
            inplanes = 320
        else:
            inplanes = 2048

        self.aspp = ASPP(backbone, output_stride)
        if use_ppm:
            self.cls = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(256, classes, kernel_size=1)
            )
        else:
            self.cls = nn.Sequential(
                nn.Conv2d(inplanes, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(256, classes, kernel_size=1)
            )


        self.memory = Memory(memory_size, feature_dim=inplanes, key_dim=inplanes, temp_update=0.1, temp_gather=0.1)

        if backbone == 'resnet':
            if output_stride == 8:
                for n, m in self.layer3.named_modules():
                    if 'conv2' in n:
                        m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                    elif 'downsample.0' in n:
                        m.stride = (1, 1)
                for n, m in self.layer4.named_modules():
                    if 'conv2' in n:
                        m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                    elif 'downsample.0' in n:
                        m.stride = (1, 1)
            elif output_stride == 16:
                for n, m in self.layer4.named_modules():
                    if 'conv2' in n:
                        m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                    elif 'downsample.0' in n:
                        m.stride = (1, 1)

            self.aux = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(256, classes, kernel_size=1)
            )

        if backbone == 'drn':
            if output_stride == 8:
                for n, m in self.layer3.named_modules():
                    if 'conv' in n:
                        m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                    elif 'downsample.0' in n:
                        m.stride = (1, 1)
                for n, m in self.layer4.named_modules():
                    if 'conv' in n:
                        m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                    elif 'downsample.0' in n:
                        m.stride = (1, 1)
            elif output_stride == 16:
                for n, m in self.layer4.named_modules():
                    if 'conv' in n:
                        m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                    elif 'downsample.0' in n:
                        m.stride = (1, 1)

            self.aux = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(256, classes, kernel_size=1)
            )

        if backbone == 'mobilenet':
            self.aux = nn.Sequential(
                nn.Conv2d(96, 96, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(96),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(96, classes, kernel_size=1)
            )

    def forward(self, x, y=None, keys=F.normalize(torch.rand((24, 2048), dtype=torch.float), dim=1).cuda()):
        x_size = x.size()

        h = int(x_size[2])
        w = int(x_size[3])

        if self.backbone == 'mobilenet':
            x1 = self.mobilenet.features[0:4](x)
            x2 = self.mobilenet.features[4:7](x1)
            x3 = self.mobilenet.features[7:14](x2)
            x4 = self.mobilenet.features[14:](x3)
        else:
            x0 = self.layer0(x)
            x1 = self.layer1(x0)
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)
            x4 = self.layer4(x3)
        aux = self.aux(x3)

        query = x4

        if self.training:
            mem_out, keys, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss = \
               self.memory(query=query, keys=keys, train=self.training)
        else:
            mem_out = self.memory(query=query, keys=keys, train=self.training)

        if self.use_ppm:
            x_spp = self.aspp(mem_out)
        else:
            x_spp = mem_out

        pred = self.cls(x_spp)

        if self.zoom_factor != 1:
            pred = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=True)

        if self.training:
            if self.zoom_factor != 1:
                aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=True)
            main_loss = self.criterion(pred, y)
            aux_loss = self.criterion(aux, y)
            return pred.max(1)[1], main_loss, aux_loss, separateness_loss, compactness_loss, keys
        else:
            return pred


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
    input = torch.rand(4, 3, 473, 473).cuda()
    model = Deeplabv3_mem(layers=50, classes=19,
                            zoom_factor=8, backbone='resnet',
                            output_stride=16).cuda()
    model.eval()
    model.training = False
    # print(model)
    # output = model(input, keys=F.normalize(torch.rand((20, 256), dtype=torch.float), dim=1).cuda(), get_feat=True, is_train_mem=False)
    # print('Deeplabv3_mem', output.size())

    from ptflops import get_model_complexity_info

    with torch.cuda.device(0):
        flops, params = get_model_complexity_info(model, (3, 688, 512), as_strings=True, print_per_layer_stat=True)

        print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
