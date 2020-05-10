"""
    Copied from https://github.com/adiyoss/GCommandsPytorch
"""

import os
import os.path

import librosa
import numpy as np
import torch
import torch.utils.data as data
from sonopy import mfcc_spec

__all__ = ['GCommand', 'GCommandMFCC']

AUDIO_EXTENSIONS = [
    '.wav', '.WAV',
]


def is_audio_file(filename):
    return any(filename.endswith(extension) for extension in AUDIO_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    spects = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_audio_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    spects.append(item)
    return spects


def spect_loader(path, window_size, window_stride, window, normalize, max_len=101):
    y, sr = librosa.load(path, sr=None)  # sr = 16000 sampling rate of `y`
    # n_fft = 4096
    n_fft = int(sr * window_size)
    win_length = n_fft
    hop_length = int(sr * window_stride)

    # STFT
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length, window=window)
    # D.shape (161, 101)
    spect, phase = librosa.magphase(D)  # D = S * P

    # S = log(S+1) Mel >>>>?
    spect = np.log1p(spect)
    # print('y.shape: {}  sr: {} spect: {}'.format(y.shape, sr, spect.shape))
    # make all spects with the same dims
    # TODO: change that in the future
    if spect.shape[1] < max_len:
        pad = np.zeros((spect.shape[0], max_len - spect.shape[1]))
        spect = np.hstack((spect, pad))
    elif spect.shape[1] > max_len:
        spect = spect[:, :max_len]
    spect = np.resize(spect, (1, spect.shape[0], spect.shape[1]))
    # spect = np.resize(spect, (1, 128, 101))
    spect = torch.FloatTensor(spect)

    # z-score normalization
    # TODO: off-line the std and mean
    if normalize:
        mean = spect.mean()
        std = spect.std()
        # print('mean: {} ,std: {}'.format(mean, std))
        if std != 0:
            spect.add_(-mean)
            spect.div_(std)
    return spect


class GCommand(data.Dataset):
    """A google command data set loader where the wavs are arranged in this way: ::
        root/one/xxx.wav
        root/one/xxy.wav
        root/one/xxz.wav
        root/head/123.wav
        root/head/nsdf3.wav
        root/head/asd932_.wav
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        window_size: window size for the stft, default value is .02
        window_stride: window stride for the stft, default value is .01
        window_type: typye of window to extract the stft, default value is 'hamming'
        normalize: boolean, whether or not to normalize the spect to have zero mean and one std
        max_len: the maximum length of frames to use
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        spects (list): List of (spects path, class_index) tuples
        STFT parameter: window_size, window_stride, window_type, normalize
    """

    def __init__(self, root, transform=None, target_transform=None, window_size=.02,
                 window_stride=.01, window_type='hamming', normalize=True, max_len=101):
        classes, class_to_idx = find_classes(root)
        spects = make_dataset(root, class_to_idx)
        if len(spects) == 0:
            raise (RuntimeError(
                "Found 0 sound files in subfolders of: " + root + "Supported audio file extensions are: " + ",".join(
                    AUDIO_EXTENSIONS)))

        self.root = root
        self.spects = spects
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = spect_loader
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_type = window_type
        self.normalize = normalize
        self.max_len = max_len

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (spect, target) where target is class_index of the target class.
        """
        path, target = self.spects[index]
        spect = self.loader(path, self.window_size, self.window_stride, self.window_type, self.normalize, self.max_len)
        if self.transform is not None:
            spect = self.transform(spect)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return spect, target

    def __len__(self):
        return len(self.spects)


class GCommandMFCC(data.Dataset):
    """A google command data set loader where the wavs are arranged in this way: ::
        root/one/xxx.wav
        root/one/xxy.wav
        root/one/xxz.wav
        root/head/123.wav
        root/head/nsdf3.wav
        root/head/asd932_.wav
    """

    def __init__(self, root, transform=None, target_transform=None,
                 windows_stride=(160, 80),
                 fft_size=512,
                 num_filt=20,
                 num_coeffs=13,
                 positive_shift=False
                 ):
        classes, class_to_idx = find_classes(root)
        spects = make_dataset(root, class_to_idx)
        if len(spects) == 0:
            raise (RuntimeError(
                "Found 0 sound files in subfolders of: " + root + "Supported audio file extensions are: " + ",".join(
                    AUDIO_EXTENSIONS)))

        self.root = root
        self.spects = spects
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = spect_loader
        self.windows_stride = windows_stride
        self.fft_size = fft_size
        self.num_filt = num_filt
        self.num_coeffs = num_coeffs
        self.max_len = int(16000 / self.windows_stride[1])
        self.positive_shift = positive_shift
        print('The mfcc spect shape: {}x{}'.format(self.max_len, self.num_coeffs))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (spect, target) where target is class_index of the target class.
        """
        path, target = self.spects[index]
        mfccs = self.get_mfcc(path)
        if self.transform is not None:
            mfccs = self.transform(mfccs)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return mfccs, target

    def get_mfcc(self, path):
        audio, sr = librosa.load(path, sr=None)  # sr = 16000 sampling rate of `y`
        mfccs = mfcc_spec(audio, sr, window_stride=self.windows_stride,
                          fft_size=self.fft_size, num_filt=self.num_filt,
                          num_coeffs=self.num_coeffs)
        mfccs /= 16
        if self.positive_shift:
            mfccs += 1
            mfccs = np.where(mfccs < 0, 0, mfccs)
        if mfccs.shape[0] < self.max_len:
            pad = np.zeros((self.max_len - mfccs.shape[0], mfccs.shape[1],))
            mfccs = np.vstack((mfccs, pad))
        elif mfccs.shape[0] > self.max_len:
            mfccs = mfccs[:, :self.max_len]
        return torch.FloatTensor(np.expand_dims(mfccs, axis=0))

    def __len__(self):
        return len(self.spects)
