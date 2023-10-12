import torch

from Utils.SignalUtils import *
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchaudio.functional as F1


class IIRFilterConv(nn.Module):
    def __init__(self, N_filt, Filt_dim, fs):
        super(IIRFilterConv, self).__init__()
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
            filters_minus = Variable(torch.zeros((self.N_filt, self.Filt_dim))).cuda()
            t_right = Variable(torch.linspace(0, N//2, steps=N//2) / self.fs).cuda()
            t_right1 = Variable((1 + torch.linspace(0, N//2, steps=N//2)) / self.fs).cuda()
        else:
            filters = Variable(torch.zeros((self.N_filt, self.Filt_dim)))
            t_right = Variable(torch.linspace(0, N//2, steps=N//2) / self.fs)
            t_right1 = Variable((1 + torch.linspace(0, N//2, steps=N//2)) / self.fs)

        min_freq = 0.0

        f1_freq = torch.abs(self.norm_f1_list) + min_freq / self.freq_scale
        f1_freq = torch.clip(f1_freq, min=0, max=0.5)
        f2_freq = f1_freq + torch.abs(self.norm_f2_list - f1_freq) + min_freq / self.freq_scale
        f2_freq = torch.clip(f2_freq, min=0, max=0.5)

        n = torch.linspace(0, N, steps=N)

        # Filter window (hamming)
        window = 0.54 - 0.46 * torch.cos(2 * torch.pi * n / N)
        if self.gpu:
            window = Variable(window.float().cuda())
        else:
            window = Variable(window.float())

        for i in range(self.N_filt):
            f1 = f1_freq[i].float() * self.freq_scale
            f2 = f2_freq[i].float() * self.freq_scale
            amp = 1
            impulse_response, _ = SignalUtils.kernel_iir(amp, f1, f2, t_right, t_right1, self.Filt_dim, self.fs)

            impulse_response = ((2 * (impulse_response - torch.min(impulse_response))) / (
                        torch.max(impulse_response) - torch.min(impulse_response) + 1e-6)) - 1
            impulse_response = (impulse_response - torch.mean(impulse_response))

            # impulse_response = impulse_response / (torch.max(impulse_response) + 1e-6)

            if self.gpu:
                filters[i, :] = impulse_response.cuda() * window
            else:
                filters[i, :] = impulse_response * window
            # filters[i, :] = impulse_response * window

        out = F.conv1d(x, filters.view(self.N_filt, 1, self.Filt_dim), groups=x.shape[1])
        return out
