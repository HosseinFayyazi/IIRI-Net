import glob
import os
import subprocess
import numpy as np
import wave
from pydub import AudioSegment
import soundfile as sf
import struct
import random


def visualize(path: str, sil=None):
    raw = wave.open(path)
    signal = raw.readframes(-1)
    signal = np.frombuffer(signal, dtype="int16")
    f_rate = raw.getframerate()
    time = np.linspace(
        0,  # start
        len(signal) / f_rate,
        num=len(signal)
    )
    if sil:
        print(path)
        new_sig = []
        start_index = 0
        for i in sil:
            if start_index != int(i[0] * f_rate):
                new_sig.append(signal[start_index: int(i[0] * f_rate)])
            start_index = int(i[1] * f_rate)
        new_sig.append(signal[start_index:])
        new_sig = np.concatenate(new_sig).ravel()
        return new_sig, f_rate
    return signal, f_rate


def detect_silence(path, time):
    command = "ffmpeg -i " + path + " -af silencedetect=n=-30dB:d=" + str(time) + " -f null -"
    out = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, stderr = out.communicate()
    s = stdout.decode("utf-8")
    k = s.split('[silencedetect @')
    if len(k) == 1:
        # print(stderr)
        return None

    start, end = [], []
    for i in range(1, len(k)):
        x = k[i].split(']')[1]
        if i % 2 == 0:
            x = x.split('|')[0]
            x = x.split(':')[1].strip()
            end.append(float(x))
        else:
            x = x.split(':')[1]
            x = x.split('size')[0]
            x = x.replace('\r', '')
            x = x.replace('\n', '').strip()
            start.append(float(x))
    return list(zip(start, end))


def create_file(rnd_train_duration, sel_file_ind, wav_file_names, sel_file_indices):
    dur = 0
    selected_signal_segment = []
    while dur < rnd_train_duration:
        sel_file_name = wav_file_names[sel_file_indices[sel_file_ind]]
        lst = detect_silence(sel_file_name, 1)
        signal, f_rate = visualize(sel_file_name, lst)
        signal = list(signal)
        sel_file_dur = len(signal) / f_rate
        # ignore below 2 seconds files
        if sel_file_dur > 2:  # get a 2 second random segment from file
            factor = 2
            if dur + 1 == rnd_train_duration:
                factor = 1
            max_rnd_ind = len(signal) - (factor * f_rate)
            rnd_start_ind = random.randint(0, max_rnd_ind)
            rnd_end_ind = rnd_start_ind + (factor * f_rate)
            segment = signal[rnd_start_ind: rnd_end_ind]
            selected_signal_segment += segment
            dur += factor
        sel_file_ind += 1
    return sel_file_ind, selected_signal_segment, f_rate


data_path = '../../Data/Librispeech/LibriSpeech/'
train_wav_file_names = glob.glob(os.path.join(data_path, "train-clean-360/**/**/*.flac"), recursive=True)
# train_wav_file_names = glob.glob(os.path.join(data_path, "test-clean/**/**/*.flac"), recursive=True)
exp_ind = '1'

# convert flac to wav
for wav_file_name in train_wav_file_names:
    os.system('ffmpeg -y -i ' + wav_file_name + ' ' + wav_file_name.replace('flac', 'wav'))
    os.remove(wav_file_name)

train_wav_file_names = glob.glob(os.path.join(data_path, "train-clean-360/**/**/*.wav"), recursive=True)
# train_wav_file_names = glob.glob(os.path.join(data_path, "test-clean/**/**/*.wav"), recursive=True)

spk_dict = {}
sil_region_dict = {}
class_lbls = set()
# build class labels
for wav_file_name in train_wav_file_names:
    speaker_id = wav_file_name.split('\\')[-3]
    class_lbls.add(speaker_id)
    spk_dict[speaker_id] = []
    sil_region_dict[speaker_id] = []
    # print(wav_file_name + ', ' + speaker_id)
class_lbls = list(class_lbls)

# build spk dict, spk_id: [file_path1, file_path2, ...]
for wav_file_name in train_wav_file_names:
    speaker_id = wav_file_name.split('\\')[-3]
    spk_dict[speaker_id].append(wav_file_name)

# determine silence regions in each file
# create train signal and save it
for spk_id in class_lbls:
    wav_file_names = spk_dict[spk_id]
    num_req_files = 0
    rnd_train_duration = random.randint(12, 15)
    rnd_val_duration = random.randint(2, 6)
    rnd_tst_duration = random.randint(2, 6)
    sel_file_indices = list(range(len(wav_file_names)))
    random.shuffle(sel_file_indices)
    # generate train segment
    sel_file_ind = 0
    sel_file_ind, selected_signal_segment, f_rate = create_file(rnd_train_duration, sel_file_ind, wav_file_names, sel_file_indices)
    sf.write(data_path + 'train' + exp_ind + '/' + spk_id + '.wav', selected_signal_segment, f_rate)
    sel_file_ind, selected_signal_segment, f_rate = create_file(rnd_val_duration, sel_file_ind, wav_file_names, sel_file_indices)
    sf.write(data_path + 'dev' + exp_ind + '/' + spk_id + '.wav', selected_signal_segment, f_rate)
    sel_file_ind, selected_signal_segment, f_rate = create_file(rnd_tst_duration, sel_file_ind, wav_file_names, sel_file_indices)
    sf.write(data_path + 'test' + exp_ind + '/' + spk_id + '.wav', selected_signal_segment, f_rate)

