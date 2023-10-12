import matplotlib.pyplot as plt

from Utils.SignalUtils import *
from Utils.GeneralUtils import *
import argparse
import matplotlib.pylab as pylab


params = {'legend.fontsize': 'x-small',
          'axes.labelsize': 'small',
          'axes.titlesize': 'small',
          'xtick.labelsize': 'x-small',
          'ytick.labelsize': 'x-small'}
pylab.rcParams.update(params)

print('Initializing parameters ...')
parser = argparse.ArgumentParser()
parser.add_argument('--out_path', type=str, default='IO/final/imgs/_epochs')

args = parser.parse_args()


print('Saving loss changes of all models in one figure ...')
model_names = [
    'Standard',
    'SincNet',
    'Sinc2Net',
    'GammaNet',
    'GaussNet',
    'IIRI-Net',
    'IIRI-Net',
    'IIRI-Net',
    'IIRI-Net'
]
model_paths = ['IO/final/cnn/1_cnn_res.res',
               'IO/final/sinc/4_kernel_sinc_res.res',
               'IO/final/sinc2/4_kernel_sinc2_res.res',
               'IO/final/gamma/1_kernel_gamma_res.res',
               'IO/final/gauss/4_kernel_gauss_res.res',
               'IO/final/iir_64/1_IIRFilter_res.res',
               'IO/final/iir_128/4_IIRFilter_res.res',
               'IO/final/iir_256/5_IIRFilter_res.res',
               'IO/final/iir_512/1_IIRFilter_res.res'
               ]
filter_lengths = [
    '0',
    '0',
    '0',
    '0',
    '0',
    '65',
    '129',
    '257',
    '512']
colors = ['k--', 'b-.', 'g-.', 'y-.', 'g:', 'm-.', 'r--', 'c-.', 'y:']
min_index = int(120 / 20)
max_index = int(220 / 20)
# min_index = int(920 / 20)
# max_index = int(1020 / 20)
for i, model_name in enumerate(model_names):
    epoch_numbers, train_losses, test_losses, fer_test, cer_test, fer_train = \
        GeneralUtils.extract_train_val_loss(model_paths[i])
    if model_name == 'IIRI-Net':
        model_name += ' (L=' + filter_lengths[i] + ')'
    # plt.plot(epoch_numbers[min_index: max_index], fer_test[min_index:max_index], colors[i], linewidth=1.2, label=model_name)
    plt.plot(epoch_numbers, fer_test, colors[i], linewidth=0.7, label=model_name)
plt.xlabel("epochs")
plt.ylabel("FER (%)")
plt.legend()
plt.savefig(args.out_path + '/' + '0_fer_test.png')
plt.close()

print('Operation completed successfully!')
