import tensorflow as tf
from scipy.signal import resample
import matplotlib.pyplot as plt
import os
import numpy as np
import glob
from keras.utils import to_categorical

# load wav 16k mono
def load_wav_16k_mono(filename):
    """load wav file as mono 16 k"""
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    wav = resample(wav.numpy(), 16000)
    return tf.convert_to_tensor(wav, dtype=tf.float32)


def audio_to_melspectogram(wav, sr=16000, n_mels=128, n_fft=2048, hop_length=512):
    """creates mel spectrogram from audio file"""
    spectrogram = tf.signal.stft(wav, frame_length=n_fft, frame_step=hop_length)
    spectrogram = tf.abs(spectrogram)
    mel_spectrogram = tf.tensordot(spectrogram, tf.signal.linear_to_mel_weight_matrix
    (num_mel_bins=n_mels, num_spectrogram_bins=spectrogram.shape[-1], sample_rate=sr), 1)

    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
    return log_mel_spectrogram


# dataset dir
data_dir = "../DATASET/sesler/Data/genres_original/"
classes = os.listdir(data_dir)

datasets = []
labels = []
for i, class_name in enumerate(classes):
    """create datasets with for loop"""
    class_path = os.path.join(data_dir, class_name)
    all_files = sorted(glob.glob(class_path+"/*.wav"))
    datasets.append(all_files)

    label = np.ones(len(all_files))*i
    labels.append(label)

data = zip(datasets, labels)


"""# concat the data
data = datasets[0]
for dataset in datasets[1:]:
    data = data.concatenate(dataset)"""


# concated preprocess function
def preprocess(file_path, label):
    wav = load_wav_16k_mono(file_path)
    spectrogram = audio_to_melspectogram(wav=wav)
    spectrogram = tf.expand_dims(spectrogram, -1)
    return spectrogram, label


spec_list = []
label_list = []
for path,lab in data:
    for p, l in zip(path, lab):
        try:
            spectogram,label = preprocess(p, l)
            spec_list.append(spectogram)
            label_list.append(to_categorical(label, num_classes=11))
        except UnicodeDecodeError:
            print("Unicode error passing")
            pass

data_spec = tf.data.Dataset.from_tensor_slices(np.array(spec_list))
data_label = tf.data.Dataset.from_tensor_slices(np.array(label_list))

data_concated = tf.data.Dataset.zip((data_spec, data_label))
data_concated.save("created")