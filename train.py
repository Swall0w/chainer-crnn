from __future__ import print_function
import argparse
import utils
import dataset
import models.crnn as crnn
import chainer
import  chainer.links as L
import  chainer.functions as F
from chainer import serializers, Variable, training
from chainer.training import extensions
import six
import numpy as np
from chainer.dataset import iterator as itr_module
from chainer import reporter


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
    opt = parser.parse_args()
    print(opt)
    return opt


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
        xs, ts = in_arrays

        optimizer = self._optimizers['main']
        xp = optimizer.target.xp
        loss_func = self.loss_func or optimizer.target

        x = Variable(np.asarray(xs)) # (64, 1, 32, 100)
        y = optimizer.target(x) # (26, 64, 37)
        padded_ts = np.zeros((len(ts), max([len(t) for t in ts])))
        for index, item in enumerate(ts):
            padded_ts[index, :item.shape[0]] = item

        loss = loss_func([item for item in y],
                         xp.asarray(padded_ts).astype(xp.int32),
                         0,
                         xp.full((len(ts),), 26, dtype=xp.int32),
                         xp.asarray([len(t) for t in ts]).astype(xp.int32))

        optimizer.target.cleargrads()
        loss.backward()
        loss.unchain_backward()
        optimizer.update()
        reporter.report({'loss': loss}, self._optimizers['main'].target)


def main():
    args = arg()

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

    convert =  utils.AlignConverter(alphabet=args.alphabet, imgH=args.imgH, imgW=args.imgW)
    # Set up a trainer
    updater = CRNNUpdater(
        train_iter, optimizer, converter=convert, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Evaluate the model with the test dataset for each epoch
#    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

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
