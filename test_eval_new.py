import pickle
import torch
import soundfile as sf
import argparse
import io
from Utils.GeneralUtils import *
from Utils.DataUtils import *
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sn


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


parser = argparse.ArgumentParser()
parser.add_argument('--save_path', type=str, default='IO/final/cnn_lib/1_saved_model.pth_best.pth')
parser.add_argument('--cfg_file', type=str, default='IO/final/cnn_lib/KernelCNN_LIB.cfg')
parser.add_argument('--dataset', type=str, default='Librispeech',
                    help='TIMIT or Librispeech')
args = parser.parse_args()

options = GeneralUtils.read_conf(args.cfg_file)

if args.dataset == 'Librispeech':
    wav_lst_tr = glob.glob(os.path.join(options['data_folder'], "train1/*.wav"), recursive=True)
    wav_lst_val = glob.glob(os.path.join(options['data_folder'], "dev1/*.wav"), recursive=True)
    wav_lst_te = glob.glob(os.path.join(options['data_folder'], "test1/*.wav"), recursive=True)
    # build label dictionary
    lbls = DataUtils.get_sid_class_labels_libri(wav_lst_tr)
else:
    wav_lst_tr = DataUtils.read_wav_file_names(options['data_folder'], train=1)
    wav_lst_tr = DataUtils.remove_sa_wav_files(wav_lst_tr)
    wav_lst_te = DataUtils.read_wav_file_names(options['data_folder'], train=0)
    wav_lst_te_comp = DataUtils.remove_sa_wav_files(wav_lst_te)
    wav_lst_te_core = GeneralUtils.get_core_wav_test_files(wav_lst_te_comp)
    wav_lst_te = wav_lst_te_comp


    wav_lst = wav_lst_tr + wav_lst_te
    lbls = DataUtils.get_sid_class_labels(wav_lst)
    wav_lst_tr, wav_lst_val, wav_lst_te = DataUtils.distinct_sid_train_test(wav_lst, lbls)


snt_tr = len(wav_lst_tr)
snt_val = len(wav_lst_val)
snt_te = len(wav_lst_te)

with open(args.save_path, 'rb') as file:
    trainer = pickle.load(file)

# with open(args.save_path, 'rb') as file:
#     trainer = CPU_Unpickler(file).load()
# trainer.gpu = False


trainer.wav_lst_te = wav_lst_te
trainer.snt_te = snt_te

trainer.CNN_net.eval()
trainer.DNN1_net.eval()
trainer.DNN2_net.eval()
test_flag = 1
loss_sum = 0
err_sum = 0
err_sum_snt = 0

predicted = []
real_lbl = []
with torch.no_grad():
    for i in range(trainer.snt_te):
        [signal, fs] = sf.read(trainer.data_folder + trainer.wav_lst_te[i])
        lab, best_class, loss, err = trainer.test_one_signal(signal, i)
        predicted.append(best_class)
        real_lbl.append(lab[0])
        err_sum_snt = err_sum_snt + (best_class != lab[0]).float()
        loss_sum = loss_sum + loss.detach()
        err_sum = err_sum + err.detach()

    err_tot_dev_snt = err_sum_snt / trainer.snt_te
    loss_tot_dev = loss_sum / trainer.snt_te
    err_tot_dev = err_sum / trainer.snt_te

print("loss_te=%f err_te=%f err_te_snt=%f" % (loss_tot_dev, err_tot_dev, err_tot_dev_snt))
print('Done!')
