import argparse
import torch
from Utils.DataUtils import *
import pickle

from SIDTrainer import *

torch.autograd.set_detect_anomaly(True)

print('Initializing parameters ...')
parser = argparse.ArgumentParser()
parser.add_argument('--cfg_file', type=str, default='IO/final/iir_128/KernelIIRFilter_TIMIT.cfg',
                    help='path of configs file')
parser.add_argument('--reduce_data', type=int, default='0', help='reduce data to run faster')
parser.add_argument('--resume_epoch', type=int, default='0', help='resume training from this epoch')
parser.add_argument('--resume_model_path', type=str, default='None',
                    help='resume training from the model with specified path')
parser.add_argument('--save_path', type=str, default='IO/final/iir_128/saved_model1.pth',
                    help='resume training from the model with specified path')
parser.add_argument('--dataset', type=str, default='TIMIT',
                    help='TIMIT or Librispeech')

args = parser.parse_args()

print('Reading options from configs file ...')
options = GeneralUtils.read_conf(args.cfg_file)

if args.resume_model_path != 'None':
    options['pt_file'] = args.resume_model_path
# setting the seed
torch.manual_seed(int(options['seed']))
np.random.seed(int(options['seed']))

print('Retrieving train and val list ..')
if args.dataset == 'Librispeech':
    wav_lst_tr = glob.glob(os.path.join(options['data_folder'], "train1/*.wav"), recursive=True)
    wav_lst_val = glob.glob(os.path.join(options['data_folder'], "dev1/*.wav"), recursive=True)
    wav_lst_te = glob.glob(os.path.join(options['data_folder'], "test1/*.wav"), recursive=True)

    snt_tr = len(wav_lst_tr)
    snt_val = len(wav_lst_val)
    snt_te = len(wav_lst_te)
    # build label dictionary
    lbls = DataUtils.get_sid_class_labels_libri(wav_lst_tr)
    lab_dict = DataUtils.build_sid_lab_dict_libri(wav_lst_tr + wav_lst_val + wav_lst_te, lbls)
    options['class_lay'] = str(len(lbls))
    print(f'The model will be trained to identify {len(lbls)} speakers.')
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

     #for test
    tr_spk_lbls = set(DataUtils.get_sid_class_labels(wav_lst_tr))
    val_spk_lbls = set(DataUtils.get_sid_class_labels(wav_lst_val))
    te_spk_lbls = set(DataUtils.get_sid_class_labels(wav_lst_te))

    options['class_lay'] = str(len(lbls))
    print(f'The model will be trained to identify {len(lbls)} speakers.')

    # build label dictionary
    lab_dict = DataUtils.build_sid_lab_dict(wav_lst, lbls)

# check folder exists
GeneralUtils.check_folder_exist(options['output_folder'])

print(f'Training {options["kernel_type"]} model ...')
freeze_first_layer = False
trainer = SIDTrainer(options, wav_lst_tr, snt_tr, wav_lst_val, snt_val, lab_dict, args.save_path, freeze_first_layer)

trainer.train(int(args.resume_epoch))
with open(args.save_path, 'wb') as filehandler:
    pickle.dump(trainer, filehandler)

print('Operation completed successfully!')


