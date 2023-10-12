from Utils.SignalUtils import *
from Utils.GeneralUtils import *
from Utils.PlotUtils import *
from Utils.DataUtils import *
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
          'ytick.labelsize': 'x-small'}
pylab.rcParams.update(params)

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='IIRFilter')
parser.add_argument('--model_path', type=str, default='IO/final/iir_128/saved_model.pth')
parser.add_argument('--cfg_file', type=str, default='IO/final/iir_128/KernelGamma_TIMIT.cfg')
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
plt.savefig(args.out_path + '_poles.png')
plt.close()


# for i in range(N_filt):
#     fig = plt.figure()
#     ax = fig.add_subplot(projection='polar')
#     w0 = 2 * np.pi * (f1_list[i] + f2_list[i]) / (2 * fs)
#     sigma = 2 * np.pi * (f2_list[i] - f1_list[i]) / (1 * fs)
#     x = np.exp(-sigma + 1j * w0)
#     print(abs(x))
#     ax.scatter(w0, abs(x), c='b', marker='x', alpha=0.5)
#     ax.scatter(-w0, abs(x), c='r', marker='x', alpha=0.5)
#     ax.scatter(-w0, 1, c='r', marker='x', alpha=0.0)
#     xT = plt.xticks()[0]
#     xL = ['0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$',
#           r'$\pi$', r'$\frac{5\pi}{4}$', r'$\frac{3\pi}{2}$', r'$\frac{7\pi}{4}$']
#     plt.xticks(xT, xL)
#     # ax.grid(False)
#     # ax.set_thetamin(0)
#     # ax.set_thetamax(180)
#     plt.savefig(args.out_path + str(i) + '_' + str(f1_list[i]) + '_' + str(f2_list[i]) + '_poles.png')
#     plt.close()

