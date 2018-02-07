from __future__ import print_function
import argparse
import random
import utils
import dataset
import models.crnn as crnn
import chainer
import  chainer.links as L
import  chainer.functions as F
from chainer.dataset import convert
from chainer import serializers, Variable, training
from chainer.training import extensions
import six
import numpy as np
from PIL import Image
from chainer.dataset import iterator as itr_module
from dataset import resizeNormalize
from skimage.transform import resize as imresize
from chainer.dataset.convert import _concat_arrays
from chainer.dataset.convert import concat_examples


def arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=2,
        help='number of data loading workers')
    parser.add_argument('--frequency', type=int, default=-1,
        help='Frequency of taking a snapshot')
    parser.add_argument('--batchsize', '-b',type=int, default=64,
        help='input batch size')
    parser.add_argument('--lexicon', default='dataset/90kDICT32px/lexicon.txt',
        type=str, help='path to lexicon file.')
    parser.add_argument('--imgH', type=int, default=32,
        help='the height of the input image to network')
    parser.add_argument('--imgW', type=int, default=100,
        help='the width of the input image to network')
    parser.add_argument('--nh', type=int, default=256,
        help='size of the lstm hidden state')
    parser.add_argument('--epoch', '-e', type=int, default=25,
        help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.01,
        help='learning rate for Critic, default=0.00005')
    parser.add_argument('--beta1', type=float, default=0.5,
        help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true',
        help='enables cuda')
    parser.add_argument('--gpu', type=int, default=-1,
        help='number of GPUs to use')
    parser.add_argument('--crnn', default='',
        help="path to crnn (to continue training)")
    parser.add_argument('--alphabet', type=str,
        default='0123456789abcdefghijklmnopqrstuvwxyz')
    parser.add_argument('--out', '-o', default='result', type=str,
        help='Where to store samples and models')
    parser.add_argument('--displayInterval', type=int, default=500,
        help='Interval to be displayed')
    parser.add_argument('--n_test_disp', type=int, default=10,
        help='Number of samples to display when test')
    parser.add_argument('--valInterval', type=int, default=500,
        help='Interval to be displayed')
    parser.add_argument('--saveInterval', type=int, default=500,
        help='Interval to be displayed')
    parser.add_argument('--adam', action='store_true',
        help='Whether to use adam (default is rmsprop)')
    parser.add_argument('--adadelta', action='store_true',
        help='Whether to use adadelta (default is rmsprop)')
    parser.add_argument('--keep_ratio', action='store_true',
        help='whether to keep ratio for image resize')
    parser.add_argument('--random_sample', action='store_true',
        help='whether to sample the dataset with random sampler')
    opt = parser.parse_args()
    print(opt)
    return opt


class AlignConverter(object):
    def __init__(self, alphabet, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio
        self.textconverter = utils.strLabelConverter(alphabet)

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

        transform = resizeNormalize((imgW, imgH))
        items = []
        for item in batchlist:
            img = transform(item[0])
            t, l = self.textconverter.encode(item[1])
            print(img.shape, item[1], t, l)
            items.append((img, t))

#        return concat_examples(items, device, )
        return variable_sequence_convert(items, device)


class CRNNUpdater(training.StandardUpdater):
    def __init__(self, iterator, optimizer, converter,
            device=None):
        if isinstance(iterator, itr_module.Iterator):
            iterator = {'main': iterator}
        self._iterators = iterator

        if not isinstance(optimizer, dict):
            optimizer = {'main': optimizer}
            self._optimizers = optimizer

        if device is not None and device >= 0:
            for optimizer in six.itervalues(self._optimizers):
                optimizer.target.to_gpu(device)

        self.converter = converter
        self.loss_func = F.connectionist_temporal_classification
        self.device = device
        self.iteration = 0

    def update_core(self):
        batch = self._iterators['main'].next()
        in_arrays = self.converter(batch, self.device)

        print('batch', type(in_arrays[0]))
        optimizer = self._optimizers['main']
        loss_func = self.loss_func or optimizer.target

        if isinstance(in_arrays, tuple):
            in_vars = tuple(Variable(x) for x in in_arrays)
            in_arrays = in_vars
        print(in_arrays)
        print(type(in_arrays))
        if isinstance(in_arrays, tuple):
            optimizer.update(loss_func, *(*in_arrays, 0))
        elif isinstance(in_arrays, dict):
            optimizer.update(loss_func, **in_arrays)
        else:
            optimizer.update(loss_func, in_arrays)


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


def main():
    args = arg()
#    if not os.path.isdir(args.out):
#        os.system('mkdir {0}'.format(args.out))

    nc = 1
    nclass = len(args.alphabet) + 1
    model = crnn.CRNN(args.imgH, nc, nclass, args.nh)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train = dataset.TextImageDataset(
        pairs_path='dataset/90kDICT32px/1ktrain.txt',
        lexicon=args.lexicon)

    test = dataset.TextImageDataset(
        pairs_path='dataset/90kDICT32px/1ktest.txt',
        lexicon=args.lexicon)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
        repeat=False, shuffle=False)

    convert =  AlignConverter(alphabet=args.alphabet, imgH=args.imgH, imgW=args.imgW)
    # Set up a trainer
    updater = CRNNUpdater(
        train_iter, optimizer, converter=convert, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

    # Take a snapshot for each specified epoch
    frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
    trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))

    trainer.extend(extensions.LogReport())

    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy',
         'validation/main/accuracy', 'elapsed_time']
         ))

    trainer.extend(extensions.ProgressBar())
    trainer.run()


if __name__ == '__main__':
    main()
