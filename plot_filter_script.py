import matplotlib.pyplot as plt
import numpy as np

from Utils.SignalUtils import *
from Utils.GeneralUtils import *
import argparse
import matplotlib.pylab as pylab
import io


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

params = {'legend.fontsize': 'x-small',
          'axes.labelsize': 'small',
          'axes.titlesize': 'small',
          'xtick.labelsize': 'x-small',
          'ytick.labelsize': 'x-small',
          'font.size': 26}
pylab.rcParams.update(params)

print('Initializing parameters ...')
parser = argparse.ArgumentParser()

parser.add_argument('--model_name', type=str, default='IIRFilter', choices=
                    ['Sinc', 'Sinc2', 'Gamma', 'Gauss', 'IIRFilter', 'CNN'])
parser.add_argument('--model_path', type=str, default='IO/final/iir_128/1_saved_model.pth_best.pth')
parser.add_argument('--cfg_file', type=str, default='IO/final/iir_128/KernelIIRFilter_TIMIT.cfg')
parser.add_argument('--out_path', type=str, default='IO/final/imgs/iir_128/')

args = parser.parse_args()

print(f'Loading {args.model_name} model ...')

with open(args.model_path, 'rb') as file:
    trainer = CPU_Unpickler(file).load()
trainer.gpu = False
model = {'CNN_model_par': trainer.CNN_net.state_dict(),
                  'DNN1_model_par': trainer.DNN1_net.state_dict(),
                  'DNN2_model_par': trainer.DNN2_net.state_dict()
                  }
# model = torch.load(args.model_path, map_location=torch.device('cpu'))
fs = 16000

options = GeneralUtils.read_conf(args.cfg_file)
N = int(list(map(int, options['cnn_len_filt'].split(',')))[0])
N_filt = int(list(map(int, options['cnn_N_filt'].split(',')))[0])
num_scales = 4

iir_h_n_s, time_domain_filters, freq_domain_filters_db, phase_of_filters, freq_centers, f1_list, f2_list, amp_list = \
    SignalUtils.get_learned_filters(args.model_name, model, fs, N, num_scales)

# check folder exists
GeneralUtils.check_folder_exist(args.out_path)

print('Saving poles in complex plane ...')
fig = plt.figure()
ax = fig.add_subplot(projection='polar')
for i in range(N_filt):
    w0 = 2 * np.pi * (f1_list[i] + f2_list[i]) / (2 * fs)
    sigma = 2 * np.pi * (f2_list[i] - f1_list[i]) / (1 * fs)
    x = np.exp(-sigma + 1j * w0)
    ax.scatter(w0, abs(x), c='b', marker='x', alpha=0.5)
    ax.scatter(-w0, abs(x), c='r', marker='x', alpha=0.5)


xT = plt.xticks()[0]
xL = ['0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$',
      r'$\pi$', r'$\frac{5\pi}{4}$', r'$\frac{3\pi}{2}$', r'$\frac{7\pi}{4}$']
plt.xticks(xT, xL)
# ax.grid(False)
# ax.set_thetamin(0)
# ax.set_thetamax(180)
plt.title('Distribution of poles of learnt filters in complex plane')
plt.savefig(args.out_path + '_poles.png', dpi=900)
plt.close()
#
# exit()

print('Saving all frequency responses of learned filters in one figure ...')
for i in range(N_filt):
    plt.plot(range(fs//2), freq_domain_filters_db[i, :], 'g', linewidth=0.2)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title(f"Freq. domain Filters learnt with {args.model_name}")
plt.savefig(args.out_path + '_freq_responses.png')
plt.close()

print('Saving All in one ...')
n_cols = 8
n_rows = int(np.ceil((N_filt * 2) / n_cols))
counter = 1
ax = plt.figure()
for i in range(N_filt):
    plt.subplot(n_rows, n_cols, counter)
    plt.plot(range(N), time_domain_filters[i, :], linewidth=0.2)
    ax.axes[counter - 1].xaxis.set_visible(False)
    ax.axes[counter - 1].yaxis.set_visible(False)
    counter += 1
    plt.subplot(n_rows, n_cols, counter)
    nf = torch.linspace(0, 0.5, steps=int(fs // 2)).numpy()
    plt.plot(nf, freq_domain_filters_db[i, :], 'g', linewidth=0.2)
    ax.axes[counter - 1].xaxis.set_visible(False)
    ax.axes[counter - 1].yaxis.set_visible(False)
    counter += 1
plt.savefig(args.out_path + '_AllInOne.png', dpi=900)
plt.close()


freq_centers = np.clip(freq_centers, a_min=0, a_max=0.5)
print('Saving the histogram of frequency centers of learned filters ...')
plt.hist(freq_centers, bins=fs//1000)
plt.xlabel("Normalized Frequency")
plt.ylabel("#filters")
plt.title(f"Center frequency histogram of {args.model_name} learned filters")
plt.savefig(args.out_path + '_freqCentersHist.png')
plt.close()


print('Saving histogram of center frequencies learnt by all models in one figure ...')
model_names = ['Mel-Scale',
               # 'Sinc',
               # 'Sinc2',
               # 'Gamma',
               # 'Gauss',
               'IIRFilter',
               'IIRFilter',
               'IIRFilter',
               'IIRFilter'
               ]
# model_labels = [
#                 'Mel-Scale',
#                'rectangular',
#                'triangular',
#                'gammatone',
#                'gaussian',
# ]
model_labels = [
                'Mel-Scale',
               'SincNet',
               'Sinc2Net',
               'GammaNet',
               'GaussNet',
]
# 'Sinc2', 'Gamma', 'Gauss']
model_paths = ['IO/final/cnn/1_saved_model.pth_best.pth',
               # 'IO/final/sinc/1_saved_model.pth_best.pth',
               # 'IO/final/sinc2/1_saved_model.pth_best.pth',
               # 'IO/final/gamma/1_saved_model.pth_best.pth',
               # 'IO/final/gauss/1_saved_model.pth_best.pth',
               'IO/final/iir_64/1_saved_model.pth_best.pth',
               'IO/final/iir_128/1_saved_model.pth_best.pth',
               'IO/final/iir_256/1_saved_model.pth_best.pth',
               'IO/final/iir_512/1_saved_model.pth_best.pth'
               ]
cfg_files = ['IO/final/cnn/KernelCNN_TIMIT.cfg',
             # 'IO/final/sinc/KernelSinc_TIMIT.cfg',
             # 'IO/final/sinc2/KernelSinc2_TIMIT.cfg',
             # 'IO/final/gamma/KernelGamma_TIMIT.cfg',
             # 'IO/final/gauss/KernelGauss_TIMIT.cfg',
             'IO/final/iir_64/KernelIIRFilter_TIMIT.cfg',
             'IO/final/iir_128/KernelIIRFilter_TIMIT.cfg',
             'IO/final/iir_256/KernelIIRFilter_TIMIT.cfg',
             'IO/final/iir_512/KernelIIRFilter_TIMIT.cfg'
             ]
filter_lengths = ['0',
                  # '0',
                  # '0',
                  # '0',
                  # '0',
                  '65',
                  '129',
                  '257',
                  '513'
                  ]
# marker = ['ko-', 'bs-', 'x', 'p', 'g>-', 'm<-', 'r^-', 'cv-', 'yD-']
marker = ['o-', 's-', 'x-', 'p-', '>-', '<-', '^-', 'v-', 'D-']
# for i, model_name in enumerate(model_names):
#     if model_name == 'Mel-Scale':
#         freq_centers = SignalUtils.get_mel_freqs(N_filt, fs)
#     else:
#         model_path = model_paths[i]
#
#         with open(model_path, 'rb') as file:
#             trainer = CPU_Unpickler(file).load()
#         trainer.gpu = False
#         model = {'CNN_model_par': trainer.CNN_net.state_dict(),
#                  'DNN1_model_par': trainer.DNN1_net.state_dict(),
#                  'DNN2_model_par': trainer.DNN2_net.state_dict()
#                  }
#
#         # model = torch.load(model_path, map_location=torch.device('cpu'))
#         cfg_file = cfg_files[i]
#         options = GeneralUtils.read_conf(cfg_file)
#         N = int(list(map(int, options['cnn_len_filt'].split(',')))[0])
#         N_filt = int(list(map(int, options['cnn_N_filt'].split(',')))[0])
#         iir_h_n_s, time_domain_filters, freq_domain_filters_db, phase_of_filters, freq_centers, f1_list, f2_list, amp_list = \
#             SignalUtils.get_learned_filters_test(model_name, model, fs, N, num_scales)
#     hist, _ = np.histogram(freq_centers, bins=fs//2000)
#     if model_name == 'IIRFilter':
#         model_name += ' (L=' + filter_lengths[i] + ')'
#     else:
#         if model_name != 'Mel-Scale':
#             model_name += 'Net'
#     plt.plot(range(1, len(hist)+1), hist, marker[i], linewidth=0.7, label=model_name)  # model_labels[i])
# plt.xlabel("Center Frequency (kHz)")
# plt.ylabel("#filters")
# plt.legend()
# plt.savefig(args.out_path + '_overal_hist.png')
# plt.close()

# exit()

iir_h_n_s, time_domain_filters, freq_domain_filters, phase_of_filters, freq_centers, f1_list, f2_list, amp_list = \
    SignalUtils.get_learned_filters(args.model_name, model, fs, N, num_scales)

print(f'Saving the images in {args.out_path} directory ...')
for i in range(N_filt):
    ax = plt.figure()
    plt.rcParams["figure.figsize"] = (10, 15)

    # plt.subplot(3, 1, 1)
    # plt.plot(range(N//2), iir_h_n_s[i, :], linewidth=1.0)
    # plt.xlabel("Samples")
    # ax.axes[0].yaxis.set_visible(False)
    # plt.title("h[n]")

    plt.subplot(2, 1, 1)
    plt.plot(range(N), time_domain_filters[i, :], linewidth=1.5)
    plt.xlabel("Samples")
    # plt.ylabel("Impulse Response")
    ax.axes[0].yaxis.set_visible(False)
    # plt.title("h[n] * h[-n]")
    # plt.title("Impulse Response")

    plt.subplot(2, 1, 2)
    freq_domain_filters[i, :] = ((40 * (freq_domain_filters[i, :] - np.min(freq_domain_filters[i, :]))) / (
            np.max(freq_domain_filters[i, :]) - np.min(freq_domain_filters[i, :]) + 1e-6)) - 20
    plt.plot(range(fs//2), freq_domain_filters[i, :], 'b', linewidth=1.5, label='Freq. Resp.')
    plt.axvline(x=f1_list[i], color='g', alpha=0.5)
    plt.axvline(x=f2_list[i], color='g', alpha=0.5)
    plt.xlabel("Frequency (Hz)")
    # plt.ylabel("Magnitude Response")
    # plt.title("Magnitude Response")
    ax.axes[1].yaxis.set_visible(False)
    # phase_of_filters[i, :] = ((20 * (phase_of_filters[i, :] - np.min(phase_of_filters[i, :]))) / (
    #         np.max(phase_of_filters[i, :]) - np.min(phase_of_filters[i, :]) + 1e-6)) - 10
    # plt.plot(range(fs // 2), phase_of_filters[i, :], 'r', linewidth=0.5, label='Scaled Phase Resp.')
    # plt.title("Freq. Response and Scaled Phase")
    # plt.legend()

    plt.savefig(args.out_path + str(i) + '.png')
    plt.close()


options = GeneralUtils.read_conf(args.cfg_file)
N = int(list(map(int, options['cnn_len_filt'].split(',')))[0])
N_filt = int(list(map(int, options['cnn_N_filt'].split(',')))[0])
iir_h_n_s, time_domain_filters, freq_domain_filters_db, phase_of_filters, freq_centers, f1_list, f2_list, amp_list = \
    SignalUtils.get_learned_filters_test(args.model_name, model, fs, N, num_scales)
for i in range(N_filt):
    plt.rcParams["figure.figsize"] = (12, 3.5)

    plt.subplot(1, 3, 1)
    if args.model_name == 'IIRFilter':
        plt.plot(range(N//2), iir_h_n_s[i, :], linewidth=0.5)
    else:
        plt.plot(range(N), iir_h_n_s[i, :], linewidth=0.5)
    plt.xlabel("Samples")
    plt.ylabel("h[n]")

    plt.subplot(1, 3, 2)
    if args.model_name == 'MsSinc':
        n_inner = int(((N - 1) / 4) + 1)
        if i // 20 == 0:
            start = (N - (n_inner - 1)) // 2
            end = start + n_inner
            plt.plot(range(n_inner), time_domain_filters[i, start:end], linewidth=0.2)
        elif i // 20 == 1:
            start = (N - 2*n_inner) // 2
            end = start + 2*n_inner - 1
            plt.plot(range(2 * (n_inner - 1) + 1), time_domain_filters[i, start:end], linewidth=0.2)
        elif i // 30 == 2:
            start = (N - 3 * n_inner) // 2
            end = start + 3 * n_inner - 2
            plt.plot(range(3 * (n_inner - 1) + 1), time_domain_filters[i, start:end], linewidth=0.2)
        else:
            plt.plot(range(N), time_domain_filters[i, :], linewidth=0.2)
    else:
        plt.plot(range(N), time_domain_filters[i, :], linewidth=0.5)
    plt.xlabel("Samples")
    if args.model_name == 'IIRFilter':
        plt.ylabel("h[n] * h[-n]")
    else:
        plt.ylabel("h[n]")

    ax1 = plt.subplot(1, 3, 3)
    if args.model_name == 'IIRFilter':
        freq_domain_filters_db[i, :] = ((40 * (freq_domain_filters_db[i, :] - np.min(freq_domain_filters_db[i, :]))) / (
                np.max(freq_domain_filters_db[i, :]) - np.min(freq_domain_filters_db[i, :]) + 1e-6)) - 20
    ax1.plot(range(fs//2), freq_domain_filters_db[i, :], 'b', linewidth=0.5)
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel('Freq. Resp. (dB)', color='k')
    if args.model_name == 'IIRFilter':
        ax1.set_ylabel('Freq. Resp. (dB)', color='b')

    if args.model_name == 'IIRFilter':
        # phase_of_filters[i, :] = ((20 * (phase_of_filters[i, :] - np.min(phase_of_filters[i, :]))) / (
        #         np.max(phase_of_filters[i, :]) - np.min(phase_of_filters[i, :]) + 1e-6)) - 10
        ax2 = ax1.twinx()
        ax2.plot(range(fs // 2), phase_of_filters[i, :], 'g', linewidth=0.5)
        ax2.set_ylabel('Unwrapped Phase Response', color='g')

    plt.subplots_adjust(wspace=0.3)
    plt.savefig(args.out_path + str(i+1) + '.png')
    plt.close()

print('Operation completed successfully!')