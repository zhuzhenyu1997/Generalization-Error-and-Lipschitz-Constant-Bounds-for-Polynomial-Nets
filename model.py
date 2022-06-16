import torch.nn as nn
import torch

# class CCP(nn.Module):
#     def __init__(self, hidden_size):
#         super(CCP, self).__init__()
#         self.image_size = 32
#         self.total_image_size = 3 * self.image_size * self.image_size
#         self.hidden_size = hidden_size
#         self.U1 = torch.nn.Linear(self.total_image_size, self.hidden_size, bias = False)
#         self.U2 = torch.nn.Linear(self.total_image_size, self.hidden_size, bias = False)
#         self.U3 = torch.nn.Linear(self.total_image_size, self.hidden_size, bias = False)
#         self.U4 = torch.nn.Linear(self.total_image_size, self.hidden_size, bias = False)
#         self.C = torch.nn.Linear(self.hidden_size, 10, bias = True)
#
#     def forward(self, z):
#         h = z.view(-1, self.total_image_size)
#         out = self.U1(h)
#         out = self.U2(h) * out + out
#         out = self.U3(h) * out + out
#         out = self.U4(h) * out + out
#         out = self.C(out)
#         return out


class CCP(nn.Module):
    def __init__(self, n_channels, n_degree=4, kernel_size=7, bias=False, downsample_degs=[2, 3], use_alpha=False, use_preconv=True):
        super(CCP, self).__init__()
        self.n_channels = n_channels
        self.image_size = 28
        before_c_size = int(self.image_size // 2 ** len(downsample_degs))
        self.total_size = before_c_size * before_c_size * n_channels
        self.n_degree = n_degree
        self.downsample_degs = downsample_degs
        self.use_alpha = use_alpha
        self.use_preconv = use_preconv
        self.n_downs = 0
        padding = int(kernel_size // 2)
        st_channels = 1
        if self.use_preconv:
            self.conv0 = nn.Conv2d(1, self.n_channels, kernel_size=3, stride=1, padding=1, bias=True)
            st_channels = self.n_channels
        for i in range(1, self.n_degree + 1):
            setattr(self, 'conv{}'.format(i), nn.Conv2d(st_channels, self.n_channels, kernel_size=kernel_size,
                                                        stride=1, padding=padding, bias=bias))
            if self.use_alpha:
                setattr(self, 'alpha{}'.format(i), nn.Parameter(torch.zeros(1)))
        self.down_sample = nn.AvgPool2d(2, 2)
        self.C = torch.nn.Linear(self.total_size, 10, bias=True)

        m = nn.GELU()
        input = torch.randn(2)
        output = m(input)

    def forward(self, z):
        if self.use_preconv:
            z = self.conv0(z)
        out = self.conv1(z)
        # # This is a flag indicating how many times the representations need to be down-sampled.
        self.n_downs = 0
        for i in range(2, self.n_degree + 1):
            temp = getattr(self, 'conv{}'.format(i))(z)
            if self.use_alpha:
                temp = getattr(self, 'alpha{}'.format(i)) * temp
            if self.n_downs > 0:
                # # down-sample appropriately before the Hadamard.
                for k in range(self.n_downs):
                    temp = self.down_sample(temp)
            out = temp * out + out
            if i in self.downsample_degs:
                out = self.down_sample(out)
                self.n_downs += 1
        out = out.view(-1, self.total_size)
        out = self.C(out)
        return out