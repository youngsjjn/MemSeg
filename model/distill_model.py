import torch
from torch import nn
import torch.nn.functional as F

import model.resnet as models
import torch.backends.cudnn as cudnn
from model.deeplabv3 import Deeplabv3
from model.sagan_models import Discriminator
from util.criterion import CriterionDSN, CriterionOhemDSN, CriterionPixelWise, \
    CriterionAdv, CriterionAdvForG, CriterionAdditionalGP, CriterionPairWiseforWholeFeatAfterPool
import util.parallel as parallel_old
from util.util import *

class Distill(nn.Module):
    def __init__(self, args):
        super(Distill, self).__init__()

        cudnn.enabled = True
        self.args = args
        device = args.device

        D_model = Discriminator(args.preprocess_GAN_mode, args.classes_num, args.batch_size, args.imsize_for_adv,
                                args.adv_conv_dim)
        load_D_model(args, D_model, False)
        print_model_parm_nums(D_model, 'D_model')
        self.parallel_D = self.DataParallelModelProcess(D_model, 2, 'train', device)

        self.D_solver = optim.SGD(
            [{'params': filter(lambda p: p.requires_grad, D_model.parameters()), 'initial_lr': args.lr_d}], args.lr_d,
            momentum=args.momentum, weight_decay=args.weight_decay)

        self.best_mean_IU = args.best_mean_IU

        self.criterion = self.DataParallelCriterionProcess(CriterionDSN())  # CriterionCrossEntropy()
        self.criterion_pixel_wise = self.DataParallelCriterionProcess(CriterionPixelWise())
        # self.criterion_pair_wise_for_interfeat = [self.DataParallelCriterionProcess(CriterionPairWiseforWholeFeatAfterPool(scale=args.pool_scale[ind], feat_ind=-(ind+1))) for ind in range(len(args.lambda_pa))]
        self.criterion_pair_wise_for_interfeat = self.DataParallelCriterionProcess(
            CriterionPairWiseforWholeFeatAfterPool(scale=args.pool_scale, feat_ind=-5))
        self.criterion_adv = self.DataParallelCriterionProcess(CriterionAdv(args.adv_loss_type))
        if args.adv_loss_type == 'wgan-gp':
            self.criterion_AdditionalGP = self.DataParallelCriterionProcess(
                CriterionAdditionalGP(self.parallel_D, args.lambda_gp))
        self.criterion_adv_for_G = self.DataParallelCriterionProcess(CriterionAdvForG(args.adv_loss_type))

        self.mc_G_loss = 0.0
        self.pi_G_loss = 0.0
        self.pa_G_loss = 0.0
        self.D_loss = 0.0

        cudnn.benchmark = True
        if not os.path.exists(args.snapshot_dir):
            os.makedirs(args.snapshot_dir)

    def forward(self, x, y=None):
        x_size = x.size()
        if y is not None:
            assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
            h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
            w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)
        else:
            h = int(x_size[2])
            w = int(x_size[3])

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_tmp = self.layer3(x)
        x = self.layer4(x_tmp)
        if self.use_ppm:
            x = self.aspp(x)
        x = self.cls(x)
        if self.zoom_factor != 1:
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        if self.training:
            aux = self.aux(x_tmp)
            if self.zoom_factor != 1:
                aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=True)
            main_loss = self.criterion(x, y)
            aux_loss = self.criterion(aux, y)
            return x.max(1)[1], main_loss, aux_loss
        else:
            return x


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
    input = torch.rand(4, 3, 473, 473).cuda()
    model = PSPNet(layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=21, zoom_factor=1, use_ppm=True, pretrained=True).cuda()
    model.eval()
    print(model)
    output = model(input)
    print('PSPNet', output.size())
