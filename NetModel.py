import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile


def conv(in_channels,
         out_channels,
         kernel_size=3,
         stride=1,
         padding=1,
         dilation=1,
         bias=False):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=bias),
        nn.ReLU(inplace=True)
    )


#########################################################################################################
# Enhanced Residual Pixel-wise Attention Block (ERPAB)
class ERPAB(nn.Module):
    def __init__(self,
                 in_channels=32,
                 mid_channels=32,
                 kernel=3,
                 stride=1,
                 d=[1, 2, 5],
                 bias=False,
                 reduction=32):
        super(ERPAB, self).__init__()

        if len(d) != 3:
            raise Exception('The length of d must match ERPAB.')

        self.inconv1 = nn.Conv2d(in_channels, mid_channels, kernel, stride, padding=d[0], dilation=d[0], bias=bias)
        self.inconv2 = nn.Conv2d(in_channels, mid_channels, kernel, stride, padding=d[1], dilation=d[1], bias=bias)
        self.inconv3 = nn.Conv2d(in_channels, mid_channels, kernel, stride, padding=d[2], dilation=d[2], bias=bias)
        self.outconv = conv(3*mid_channels, in_channels, kernel_size=1, stride=stride, padding=0, bias=True)

        self.pa = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels//reduction, kernel, stride, padding=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels//reduction, mid_channels, kernel, stride, padding=1, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):

        input_ = x
        x0 = self.inconv1(x)
        x1 = self.inconv2(x)
        x2 = self.inconv3(x)
        x = torch.cat((x0, x1, x2), 1)
        x = self.outconv(x)
        res = self.pa(x)
        x = F.relu(x * res + input_)

        return x


############################################################################################################
# Dilated Dense Residual Block (DDRB)
class DDRB(nn.Module):
    def __init__(self,
                 in_channels=32,
                 mid_channels=32,
                 kernel=3,
                 stride=1,
                 d=[1, 1, 2, 2, 5, 5],
                 bias=False):
        super(DDRB, self).__init__()

        if len(d) != 6:
            raise Exception('The length of d must match DDRB.')

        self.dconv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel, stride, padding=d[0], dilation=d[0], bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel, stride, padding=d[1], dilation=d[1], bias=bias),
        )
        self.dconv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel, stride, padding=d[2], dilation=d[2], bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel, stride, padding=d[3], dilation=d[3], bias=bias)
        )
        self.dconv5 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel, stride, padding=d[4], dilation=d[4], bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel, stride, padding=d[5], dilation=d[5], bias=bias)
        )

    def forward(self, x):
        x1 = self.dconv1(x)
        x2 = self.dconv2(F.relu(x + x1))
        x3 = self.dconv5(F.relu(x + x1 + x2))
        x = F.relu(x + x1 + x2 + x3)
        return x


#########################################################################################
# Light Dual-stage Progressive Enhancement Network (LightDPENet)
class LightDPENet(nn.Module):
    def __init__(self,
                 in_channels=3,
                 mid_channels=32,
                 kernel=3,
                 stride=1,
                 dilation_ddrb=[1, 1, 2, 2, 5, 5],
                 dialtion_erpab=[1, 2, 5],
                 n_ddrb=5,
                 n_erpab=3,
                 bias=False):
        super(DPENet, self).__init__()

        self.inconv1 = conv(in_channels, mid_channels, kernel_size=1, padding=0, bias=bias)
        self.inconv2 = conv(in_channels, mid_channels, kernel_size=1, padding=0, bias=bias)
        self.outconv1 = conv(mid_channels, in_channels, kernel_size=1, padding=0, bias=bias)
        self.outconv2 = conv(mid_channels, in_channels, kernel_size=1, padding=0, bias=bias)

        self.ddrb1 = DDRB(mid_channels, mid_channels, kernel, stride, dilation_ddrb, bias)
        self.ddrb2 = DDRB(mid_channels, mid_channels, kernel, stride, dilation_ddrb, bias)
        self.ddrb3 = DDRB(mid_channels, mid_channels, kernel, stride, dilation_ddrb, bias)
        self.ddrb4 = DDRB(mid_channels, mid_channels, kernel, stride, dilation_ddrb, bias)
        self.ddrb5 = DDRB(mid_channels, mid_channels, kernel, stride, dilation_ddrb, bias)

        self.erpab1 = ERPAB(mid_channels, mid_channels, kernel, stride, dialtion_erpab, bias)
        self.erpab2 = ERPAB(mid_channels, mid_channels, kernel, stride, dialtion_erpab, bias)
        self.erpab3 = ERPAB(mid_channels, mid_channels, kernel, stride, dialtion_erpab, bias)

    def forward(self, x):
        input_ = x

        # Rain Streaks Removal Network (R2Net)
        x = self.inconv1(x)
        x = self.ddrb1(x)
        x = self.ddrb2(x)
        x = self.ddrb3(x)
        x = self.ddrb4(x)
        x = self.ddrb5(x)
        x = self.outconv1(x)
        x_mid = x + input_

        # Detail Reconstruction Network (DRNet)
        x = self.inconv2(F.relu(x_mid))
        x = self.erpab1(x)
        x = self.erpab2(x)
        x = self.erpab3(x)
        x = self.outconv2(x)
        x = x + x_mid

        return x_mid, x



#########################################################################################
# Dual-stage Progressive Enhancement Network (DPENet)
class DPENet(nn.Module):
    def __init__(self,
                 in_channels=3,
                 mid_channels=32,
                 kernel=3,
                 stride=1,
                 dilation_ddrb=[1, 1, 2, 2, 5, 5],
                 dialtion_erpab=[1, 2, 5],
                 n_ddrb=10,
                 n_erpab=3,
                 bias=False):
        super(DPENet, self).__init__()

        self.inconv1 = conv(in_channels, mid_channels, kernel_size=1, padding=0, bias=bias)
        self.inconv2 = conv(in_channels, mid_channels, kernel_size=1, padding=0, bias=bias)
        self.outconv1 = conv(mid_channels, in_channels, kernel_size=1, padding=0, bias=bias)
        self.outconv2 = conv(mid_channels, in_channels, kernel_size=1, padding=0, bias=bias)

        self.ddrb1 = DDRB(mid_channels, mid_channels, kernel, stride, dilation_ddrb, bias)
        self.ddrb2 = DDRB(mid_channels, mid_channels, kernel, stride, dilation_ddrb, bias)
        self.ddrb3 = DDRB(mid_channels, mid_channels, kernel, stride, dilation_ddrb, bias)
        self.ddrb4 = DDRB(mid_channels, mid_channels, kernel, stride, dilation_ddrb, bias)
        self.ddrb5 = DDRB(mid_channels, mid_channels, kernel, stride, dilation_ddrb, bias)
        self.ddrb6 = DDRB(mid_channels, mid_channels, kernel, stride, dilation_ddrb, bias)
        self.ddrb7 = DDRB(mid_channels, mid_channels, kernel, stride, dilation_ddrb, bias)
        self.ddrb8 = DDRB(mid_channels, mid_channels, kernel, stride, dilation_ddrb, bias)
        self.ddrb9 = DDRB(mid_channels, mid_channels, kernel, stride, dilation_ddrb, bias)
        self.ddrb10 = DDRB(mid_channels, mid_channels, kernel, stride, dilation_ddrb, bias)

        self.erpab1 = ERPAB(mid_channels, mid_channels, kernel, stride, dialtion_erpab, bias)
        self.erpab2 = ERPAB(mid_channels, mid_channels, kernel, stride, dialtion_erpab, bias)
        self.erpab3 = ERPAB(mid_channels, mid_channels, kernel, stride, dialtion_erpab, bias)

    def forward(self, x):
        input_ = x

        # Rain Streaks Removal Network (R2Net)
        x = self.inconv1(x)
        x = self.ddrb1(x)
        x = self.ddrb2(x)
        x = self.ddrb3(x)
        x = self.ddrb4(x)
        x = self.ddrb5(x)
        x = self.ddrb6(x)
        x = self.ddrb7(x)
        x = self.ddrb8(x)
        x = self.ddrb9(x)
        x = self.ddrb10(x)
        x = self.outconv1(x)
        x_mid = x + input_

        # Detail Reconstruction Network (DRNet)
        x = self.inconv2(F.relu(x_mid))
        x = self.erpab1(x)
        x = self.erpab2(x)
        x = self.erpab3(x)
        x = self.outconv2(x)
        x = x + x_mid

        return x_mid, x


############################################################
# Network Demo
if __name__ == '__main__':

    # myNet = LightDPENet()
    myNet = DPENet()
    myNet.load_state_dict(torch.load('./logs/Rain800.pth'))
    myNet = myNet.cuda()
    t = 0
    for i in range(100):
        x = torch.randn(1, 3, 256, 256)
        x = x.cuda()
        t0 = time.time()
        out = myNet(x)
        t1 = time.time()
        t += t1 - t0
    print(f'The average spending time is {t / 100} s.')
    flops, params = profile(myNet, inputs=(x, ))
    print(f'flops:{flops / (1000 ** 3)}, params:{params / (1000 ** 2)}')

