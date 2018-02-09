#!/usr/bin/python
# encoding: utf-8

import chainer
from chainer import Variable
import collections
import numpy as np
from dataset import resizeNormalize


class strLabelConverter(object):
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case=True):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'  # for `-1` index

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank'
            self.dict[char] = i + 1

    def encode(self, text):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            numpy array [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            numpy array [n]: length of each text.
        """
        if isinstance(text, str):
            text = [
                self.dict[char.lower() if self._ignore_case else char]
                for char in text
            ]
            length = [len(text)]
        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        return (np.array(text).astype(np.int8), np.array(length).astype(np.int8))

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.

        Args:
            numpy array [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            numpy array [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if length.size == 1:
            length = length[0]
            assert t.size == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.size == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.size):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], Variable(np.array([l])).data, raw=raw))
                index += l
            return texts


class AlignConverter(object):
    def __init__(self, alphabet, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio
        self.textconverter = strLabelConverter(alphabet)

    def __call__(self, batch, device=None):
        if len(batch) == 0:
            raise ValueError('batch is empty')

        batchlist = batch
        imgH = self.imgH
        imgW = self.imgW
        if self.keep_ratio:
            ratios = []
            for item in batchlist:
                w, h = item[0].shape
                ratios.append(w / float(h))
            ratios.sort()
            max_ratio = ratios[-1]
            imgW = int(np.float(max_ratio * imgH))
            imgW = max(imgH * self.min_ratio, imgW)

        transform = resizeNormalize((imgH, imgW))
        items = []
        for item in batchlist:
            img = transform(item[0])
            t, l = self.textconverter.encode(item[1])
            items.append((img, t))

        return variable_sequence_convert(items, device)


def variable_sequence_convert(batch, device):
    if device is None:
        def to_device(x):
            return x
    elif device < 0:
        to_device = chainer.cuda.to_cpu
    else:
        def to_device(x):
            return chainer.cuda.to_gpu(x, device, chainer.cuda.Stream.null)

    def to_device_batch(batch):
        if device is None:
            return batch
        elif device < 0:
            return [to_device(x) for x in batch]
        else:
            xp = chainer.cuda.cupy.get_array_module(*batch)
            concat = xp.concatenate(batch, axis=0)
            sections = np.cumsum([len(x) for x in batch[:-1]], dtype='i')
            concat_dev = to_device(concat)
            batch_dev = chainer.cuda.cupy.split(concat_dev, sections)
            return batch_dev

    return tuple([to_device_batch([x for x, _ in batch]), to_device_batch([y for _, y in batch])])
