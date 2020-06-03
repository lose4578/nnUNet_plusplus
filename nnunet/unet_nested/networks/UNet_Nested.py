import torch
import torch.nn as nn
from nnunet.unet_nested.layers import unetConv2, unetUp
from nnunet.unet_nested.utils import init_weights, count_param
from nnunet.config import Config


# class UNet_Nested(nn.Module):
#
#     def __init__(self, in_channels=1, n_classes=1, feature_scale=2, is_deconv=False, is_batchnorm=True, is_ds=True):
#         is_deconv = False
#         super(UNet_Nested, self).__init__()
#         self.in_channels = in_channels
#         self.feature_scale = feature_scale
#         self.is_deconv = is_deconv
#         self.is_batchnorm = is_batchnorm
#         self.is_ds = is_ds
#
#         filters = [64, 128, 256, 512, 1024]
#         filters = [int(x / self.feature_scale) for x in filters]
#
#         # downsampling
#         self.maxpool = nn.MaxPool3d(kernel_size=2)
#         # self.conv_in =
#         self.conv00 = unetConv2(self.in_channels, filters[0], self.is_batchnorm, alpha_in=0)
#         self.conv10 = unetConv2(filters[0], filters[1], self.is_batchnorm)
#         self.conv20 = unetConv2(filters[1], filters[2], self.is_batchnorm)
#         self.conv30 = unetConv2(filters[2], filters[3], self.is_batchnorm)
#         # self.conv40 = unetConv2(filters[3], filters[4], self.is_batchnorm, alpha_in=0.5, alpha_out=0)
#
#         # upsampling
#         self.up_concat01 = unetUp(filters[1], filters[0], self.is_deconv)
#         self.up_concat11 = unetUp(filters[2], filters[1], self.is_deconv)
#         self.up_concat21 = unetUp(filters[3], filters[2], self.is_deconv)
#         # self.up_concat31 = unetUp(filters[4], filters[3], self.is_deconv)
#
#         self.up_concat02 = unetUp(filters[1], filters[0], self.is_deconv, 3)
#         self.up_concat12 = unetUp(filters[2], filters[1], self.is_deconv, 3)
#         # self.up_concat22 = unetUp(filters[3], filters[2], self.is_deconv, 3)
#
#         self.up_concat03 = unetUp(filters[1], filters[0], self.is_deconv, 4)
#         # self.up_concat13 = unetUp(filters[2], filters[1], self.is_deconv, 4)
#
#         # self.up_concat04 = unetUp(filters[1], filters[0], self.is_deconv, 5)
#
#         # final conv (without any concat)
#         self.final_1 = OctConv_3D(filters[0], n_classes, kernel_size=1, stride=1, alpha_in=0.5, alpha_out=0)
#         self.final_2 = OctConv_3D(filters[0], n_classes, kernel_size=1, stride=1, alpha_in=0.5, alpha_out=0)
#         self.final_3 = OctConv_3D(filters[0], n_classes, kernel_size=1, stride=1, alpha_in=0.5, alpha_out=0)
#         # self.final_4 = OctConv_3D(filters[0], n_classes, kernel_size=1)
#         # self.final_1 = OctConv_3D(filters[0], n_classes, kernel_size=1)
#         # self.final_2 = nn.Conv3d(filters[0], n_classes, 1)
#         # self.final_3 = nn.Conv3d(filters[0], n_classes, 1)
#         # self.final_4 = nn.Conv3d(filters[0], n_classes, 1)
#
#         # initialise weights
#         for m in self.modules():
#             if isinstance(m, nn.Conv3d):
#                 init_weights(m, init_type='kaiming')
#             elif isinstance(m, nn.BatchNorm3d):
#                 init_weights(m, init_type='kaiming')
#
#     def forward(self, inputs):
#         # column : 0
#         X_00_h, X_00_l = self.conv00(inputs)  # 16*512*512
#         maxpool0_h = self.maxpool(X_00_h)
#         maxpool0_l = self.maxpool(X_00_l)  # 16*256*256
#         X_10_h, X_10_l = self.conv10((maxpool0_h, maxpool0_l))  # 32*256*256
#         maxpool1_h = self.maxpool(X_10_h)
#         maxpool1_l = self.maxpool(X_10_l)  # 32*128*128
#         X_20_h, X_20_l = self.conv20((maxpool1_h, maxpool1_l))  # 64*128*128
#         maxpool2_h = self.maxpool(X_20_h)  # 64*64*64
#         maxpool2_l = self.maxpool(X_20_l)
#         X_30_h, X_30_l = self.conv30((maxpool2_h, maxpool2_l))  # 128*64*64
#         # maxpool3_h = self.maxpool(X_30_h)  # 128*32*32
#         # maxpool3_l = self.maxpool(X_30_l)
#         # # pdb.set_trace()
#         # X_40_h, X_40_l = self.conv40((maxpool3_h, maxpool3_l))  # 256*32*32
#         # column : 1
#         X_01_h, X_01_l = self.up_concat01((X_10_h, X_10_l), (X_00_h,), (X_00_l,))
#         X_11_h, X_11_l = self.up_concat11((X_20_h, X_20_l), (X_10_h,), (X_10_l,))
#         X_21_h, X_21_l = self.up_concat21((X_30_h, X_30_l), (X_20_h,), (X_20_l,))
#         # X_31 = self.up_concat31(X_40_h, X_30_h)
#         # column : 2
#         X_02_h, X_02_l = self.up_concat02((X_11_h, X_11_l), (X_00_h, X_01_h), (X_00_l, X_01_l))
#         X_12_h, X_12_l = self.up_concat12((X_21_h, X_21_l), (X_10_h, X_11_h), (X_10_l, X_11_l))
#         # X_22 = self.up_concat22(X_31, X_20_h, X_21)
#         # column : 3
#         X_03_h, X_03_l = self.up_concat03((X_12_h, X_12_l), (X_00_h, X_01_h, X_02_h), (X_00_l, X_01_l, X_02_l))
#         # X_13 = self.up_concat13(X_22, X_10_h, X_11, X_12)
#         # column : 4
#         # X_04 = self.up_concat04(X_13, X_00_h, X_01, X_02, X_03)
#
#         # final layer
#         final_1_h, final_1_l = self.final_1((X_01_h, X_01_l))
#         final_2_h, final_2_l = self.final_2((X_02_h, X_02_l))
#         final_3_h, final_3_l = self.final_3((X_03_h, X_03_l))
#         # final_4 = self.final_4(X_04)
#
#         # final = (final_1 + final_2 + final_3 + final_4) / 4
#         final = (final_1_h + final_2_h + final_3_h) / 3
#
#         if self.is_ds:
#             return final
#         else:
#             return final_3_h


class UNet_Nested(nn.Module):

    def __init__(self, in_channels=1, n_classes=2, feature_scale=4, is_deconv=True, is_batchnorm=True, is_ds=True):
        super(UNet_Nested, self).__init__()
        self.in_channels = in_channels
        self.feature_scale = feature_scale
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        self.is_ds = is_ds

        # f_size = 52
        # filters = [f_size, f_size * 2, f_size * 4, f_size * 8, f_size * 16]
        # filters = [int(x / self.feature_scale) for x in filters]

        # f_size = 16
        # filters = [f_size, f_size+56, f_size+(56*2), f_size+(56*3), f_size+(56*4)]

        f_size = 14
        filters = [f_size, f_size+40, f_size+(40*2), f_size+(40*3), f_size+(40*4)]

        # downsampling
        self.dropout = nn.Dropout(p=0.2)
        self.maxpool = nn.AvgPool3d(kernel_size=2)
        self.conv00 = unetConv2(
            self.in_channels, filters[0], self.is_batchnorm)
        self.conv10 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.conv20 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.conv30 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.conv40 = unetConv2(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat01 = unetUp(filters[1], filters[0], self.is_deconv)
        self.up_concat11 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat21 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat31 = unetUp(filters[4], filters[3], self.is_deconv)

        self.up_concat02 = unetUp(filters[1], filters[0], self.is_deconv, 3)
        self.up_concat12 = unetUp(filters[2], filters[1], self.is_deconv, 3)
        self.up_concat22 = unetUp(filters[3], filters[2], self.is_deconv, 3)

        self.up_concat03 = unetUp(filters[1], filters[0], self.is_deconv, 4)
        self.up_concat13 = unetUp(filters[2], filters[1], self.is_deconv, 4)

        self.up_concat04 = unetUp(filters[1], filters[0], self.is_deconv, 5)

        # final conv (without any concat)
        self.final_1 = nn.Conv3d(filters[0], n_classes, 1)
        self.final_2 = nn.Conv3d(filters[0], n_classes, 1)
        self.final_3 = nn.Conv3d(filters[0], n_classes, 1)
        self.final_4 = nn.Conv3d(filters[0], n_classes, 1)

        self.config = Config()

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        # column : 0
        X_00 = self.conv00(inputs)  # 16*512*512
        maxpool0 = self.maxpool(X_00)  # 16*256*256
        # X_00 = self.dropout(X_00)
        X_10 = self.conv10(maxpool0)  # 32*256*256
        maxpool1 = self.maxpool(X_10)  # 32*128*128
        # X_10 = self.dropout(X_10)
        X_20 = self.conv20(maxpool1)  # 64*128*128
        maxpool2 = self.maxpool(X_20)  # 64*64*64
        # X_20 = self.dropout(X_20)
        X_30 = self.conv30(maxpool2)  # 128*64*64
        maxpool3 = self.maxpool(X_30)  # 128*32*32
        # X_30 = self.dropout(X_30)
        X_40 = self.conv40(maxpool3)  # 256*32*32
        # X_40 = self.dropout(X_40)
        # column : 1
        X_01 = self.up_concat01(X_10, X_00)
        X_11 = self.up_concat11(X_20, X_10)
        X_21 = self.up_concat21(X_30, X_20)
        X_31 = self.up_concat31(X_40, X_30)
        # column : 2
        X_02 = self.up_concat02(X_11, X_00, X_01)
        X_12 = self.up_concat12(X_21, X_10, X_11)
        X_22 = self.up_concat22(X_31, X_20, X_21)
        # column : 3
        X_03 = self.up_concat03(X_12, X_00, X_01, X_02)
        X_13 = self.up_concat13(X_22, X_10, X_11, X_12)
        # column : 4
        X_04 = self.up_concat04(X_13, X_00, X_01, X_02, X_03)

        # final layer
        final_1 = self.final_1(X_01)
        # final_1 = self.dropout(final_1)
        final_2 = self.final_2(X_02)
        # final_2 = self.dropout(final_2)
        final_3 = self.final_3(X_03)
        # final_3 = self.dropout(final_3)
        final_4 = self.final_4(X_04)
        # final_4 = self.dropout(final_4)

        # plot_kernels(final_4[0].cpu(), 2)

        # final = (0.28*final_1 + 0.28*final_2 + 0.28*final_3 + 0.16*final_4)
        final = (final_1 + final_2 + final_3 + final_4) / 4
        # final = torch.add(0.28*final_1, 0.28*final_2)
        # final = torch.add(final, 0.28*final_3)
        # final = torch.add(final, 0.16*final_4)
        # final = torch.div(final, 4)

        if self.config.debug != None:
            print('debug:', self.config.debug + 1, '......')
            return [final_1, final_2, final_3, final_4][self.config.debug]
        if self.is_ds:
            return final
        else:
            return final


if __name__ == '__main__':
    print('#### Test Case ###')
    from torch.autograd import Variable

    x = Variable(torch.rand(2, 1, 64, 64)).cuda()
    model = UNet_Nested().cuda()
    param = count_param(model)
    y = model(x)
    print('Output shape:', y.shape)
    print('unet_nested totoal parameters: %.2fM (%d)' % (param / 1e6, param))
