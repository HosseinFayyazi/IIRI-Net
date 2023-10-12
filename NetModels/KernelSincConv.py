from Utils.SignalUtils import *
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class KernelSincConv(nn.Module):
    def __init__(self, N_filt, Filt_dim, fs):
        super(KernelSincConv, self).__init__()
        self.gpu = torch.cuda.is_available()
        # Mel Initialization of the filterbanks
        mel_freqs = SignalUtils.get_mel_freqs(N_filt, fs)
        f1, f2 = SignalUtils.get_f1_f2(mel_freqs, fs)

        self.freq_scale = fs * 1.0
        self.norm_f1_list = nn.Parameter(torch.from_numpy(f1 / self.freq_scale))
        self.norm_f2_list = nn.Parameter(torch.from_numpy(f2 / self.freq_scale))

        self.N_filt = N_filt
        self.Filt_dim = Filt_dim
        self.fs = fs

    def forward(self, x):
        N = self.Filt_dim
        if self.gpu:
            filters = Variable(torch.zeros((self.N_filt, self.Filt_dim))).cuda()
            t_right = Variable(torch.linspace(1, (N - 1) / 2, steps=int((N - 1) / 2)) / self.fs).cuda()
        else:
            filters = Variable(torch.zeros((self.N_filt, self.Filt_dim)))
            t_right = Variable(torch.linspace(1, (N - 1) / 2, steps=int((N - 1) / 2)) / self.fs)



        min_freq = 50.0
        min_band = 50.0

        f1_freq = torch.abs(self.norm_f1_list) + min_freq / self.freq_scale
        f2_freq = f1_freq + torch.abs(self.norm_f2_list - f1_freq) + min_freq / self.freq_scale

        n = torch.linspace(0, N, steps=N)

        # Filter window (hamming)
        window = 0.54 - 0.46 * torch.cos(2 * torch.pi * n / N)
        if self.gpu:
            window = Variable(window.float().cuda())
        else:
            window = Variable(window.float())

        for i in range(self.N_filt):
            f1 = f1_freq[i].float() * self.fs
            f2 = f2_freq[i].float() * self.fs
            impulse_response = SignalUtils.kernel_sinc(f1, f2, t_right)
            impulse_response = impulse_response / torch.max(impulse_response)
            if self.gpu:
                filters[i, :] = impulse_response.cuda() * window
            else:
                filters[i, :] = impulse_response * window
        out = F.conv1d(x, filters.view(self.N_filt, 1, self.Filt_dim))
        return out