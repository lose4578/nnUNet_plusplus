import torch
import torch.nn as nn
from unet_nested.utils import init_weights
from octconv import OctConv_ACT_3D, OctConv_BN_ACT_3D, OctConv_3D
import pdb


# class unetConv2(nn.Module):
#     def __init__(self, in_size, out_size, is_batchnorm, alpha_in=None, alpha_out=None, n=2, ks=3, stride=1, padding=1, ):
#         super(unetConv2, self).__init__()
#         self.n = n
#         self.ks = ks
#         self.stride = stride
#         self.padding = padding
#         s = stride
#         p = padding
#         if is_batchnorm:
#             for i in range(1, n + 1):
#                 # conv = nn.Sequential(Conv_BN_3D(in_size, out_size, ks, s, p),
#                 #                      # nn.BatchNorm3d(out_size),
#                 #                      nn.ReLU(inplace=True), )
#                 if (alpha_in != None) and i == 1:
#                     conv = OctConv_BN_ACT_3D(in_size, out_size, kernel_size=ks, stride=s, padding=p, alpha_in=alpha_in)
#                 elif (alpha_out != None) and i == n:
#                     conv = OctConv_BN_ACT_3D(in_size, out_size, kernel_size=ks, stride=s, padding=p, alpha_in=alpha_in, alpha_out=alpha_out)
#                 else:
#                     conv = OctConv_BN_ACT_3D(in_size, out_size, kernel_size=ks, stride=s, padding=p)
#                 setattr(self, 'conv%d' % i, conv)
#                 in_size = out_size
#
#         else:
#             for i in range(1, n + 1):
#                 # conv = nn.Sequential(OctaveConv_3D(in_size, out_size, ks, s, p),
#                 #                      nn.ReLU(inplace=True), )
#                 if (alpha_in != None) and i == 1:
#                     conv = OctConv_ACT_3D(in_size, out_size, kernel_size=ks, stride=s, padding=p, alpha_in=alpha_in)
#                 elif (alpha_out != None) and i == n:
#                     conv = OctConv_ACT_3D(in_size, out_size, kernel_size=ks, stride=s, padding=p, alpha_in=alpha_in, alpha_out=alpha_out)
#                 else:
#                     conv = OctConv_ACT_3D(in_size, out_size, kernel_size=ks, stride=s, padding=p)
#                 # conv = OctConv_ACT_3D(in_size, out_size, ks, s, p)
#                 setattr(self, 'conv%d' % i, conv)
#                 in_size = out_size
#
#         # initialise the blocks
#         for m in self.children():
#             init_weights(m, init_type='kaiming')
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv3d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#     def forward(self, inputs):
#         x = inputs
#         x_h, x_l = x if type(x) is tuple else (x, None)
#         for i in range(1, self.n + 1):
#             conv = getattr(self, 'conv%d' % i)
#             x_h, x_l = conv((x_h, x_l))
#
#         return x_h, x_l
#
#
# class unetUp(nn.Module):
#     def __init__(self, in_size, out_size, is_deconv, n_concat=2, alpha_in=None, alpha_out=None):
#         super(unetUp, self).__init__()
#
#         if alpha_in != None:
#             pass
#         else:
#             alpha_in = 0.5
#
#         if alpha_out != None:
#             pass
#         else:
#             alpha_out = 0.5
#
#         self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
#         if is_deconv:
#             self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size=2, stride=2, padding=0)
#         else:
#             # self.up = nn.Sequential(
#             #     nn.UpsamplingBilinear2d(scale_factor=2),
#             #     OctConv_3D(in_size, out_size, kernel_size=3, alpha_in=alpha_in, alpha_out=alpha_out))
#             self.upsampling = nn.Upsample(scale_factor=2)
#             self.OctConv_3D = OctConv_3D(in_size, out_size, kernel_size=3, alpha_in=alpha_in, alpha_out=alpha_out, padding=1)
#
#         # initialise the blocks
#         # for m in self.children():
#         #     if m.__class__.__name__.find('unetConv2') != -1: continue
#         #     init_weights(m, init_type='kaiming')
#         for m in self.modules():
#             if isinstance(m, nn.Conv3d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#     def forward(self, high_feature, low_feature_h, low_feature_l):
#         # pdb.set_trace()
#         outputs0_h, outputs0_l = high_feature
#         outputs0_h, outputs0_l = self.OctConv_3D((outputs0_h, outputs0_l))
#         outputs0_h = self.upsampling(outputs0_h)
#         outputs0_l = self.upsampling(outputs0_l)
#
#
#         for feature in low_feature_h:
#             outputs0_h = torch.cat([outputs0_h, feature], 1)
#
#         for feature in low_feature_l:
#             outputs0_l = torch.cat([outputs0_l, feature], 1)
#
#         return self.conv((outputs0_h, outputs0_l))


class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv3d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm3d(out_size),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv3d(in_size, out_size, ks, s, p),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')
        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)
        # x = self.dropout(x)
        return x


class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2):
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size=2, stride=2, padding=0)
        else:
            self.up = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv3d(in_size, out_size, kernel_size=()))

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1: continue
            init_weights(m, init_type='kaiming')
        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, high_feature, *low_feature):
        outputs0 = self.up(high_feature)
        for feature in low_feature:
            outputs0 = torch.cat([outputs0, feature], 1)
        outputs0 = self.conv(outputs0)
        # outputs0 = self.dropout(outputs0)
        return outputs0
