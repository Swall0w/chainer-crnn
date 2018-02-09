#!/usr/bin/python
# encoding: utf-8

import random
import six
import numpy as np
from skimage.transform import resize as imresize
import chainer
import os
import skimage.io as skio


class resizeNormalize(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        # image shape should be (ch, h, w) 0 <= pix <= 1
        if len(img.shape) == 2:
            img = img[np.newaxis, :]
        img = np.transpose(img, (1, 2, 0))
        resized_image = imresize(img, self.size, mode='reflect')
        resized_image = resized_image.transpose(2, 0, 1).astype(np.float32)
        img = resized_image - 0.5
        return img


class TextImageDataset(chainer.dataset.DatasetMixin):
    def __init__(self, pairs_path, lexicon, label_dict=None, dtype=np.float32,
            label_dtype=np.int32, resize=None, random_step=0):
        self.path_to_target_txt = '{}/'.format(os.path.split(pairs_path)[0])
        if isinstance(pairs_path, six.string_types):
            with open(pairs_path) as pairs_file:
                pairs = []
                for i, line in enumerate(pairs_file):
                    pair = line.strip().split()
                    if len(pair) != 2:
                        raise ValueError(
                                'invalid format at line {} in file {}'.format(
                                    i, pairs_path))
                    pairs.append((pair[0], str(pair[1])))

        if isinstance(lexicon, six.string_types):
            l_names = []
            with open(lexicon) as lexicon_file:
                for i, line in enumerate(lexicon_file):
                    name = line.strip().split()
                    if len(name) != 1:
                        raise ValueError('invalid format')
                    l_names.append(str(*name))

        self._lexicon = l_names
        self._pairs = pairs
        self._dtype = dtype
        self._label_dtype = label_dtype
        self.resize = resize
        self.label_dict = label_dict

    def __len__(self):
        return len(self._pairs)

    def get_example(self, i):
        img_path, label = self._pairs[i]
        full_path = os.path.abspath(self.path_to_target_txt + img_path)
        image = _read_image_as_array(full_path, self._dtype)
        if len(image.shape) == 2:
            image = image[np.newaxis, :]

        text = self._lexicon[int(label)]
        return image, text


def _read_image_as_array(path, dtype):
    image = skio.imread(path, as_grey=True)
    image = np.expand_dims(image, axis=0)
    image = np.asarray(image, dtype=dtype)

    return image
