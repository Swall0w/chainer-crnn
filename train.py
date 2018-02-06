from __future__ import print_function
import argparse
import random
import os
import utils
import dataset

import models.crnn as crnn


import chainer
import  chainer.links as L
from chainer.dataset import convert
from chainer import serializers
from chainer import Variable
import six
import numpy as np
from PIL import Image


def arg():
    parser = argparse.ArgumentParser()
#    parser.add_argument('--trainroot', required=True,
#        help='path to dataset')
#    parser.add_argument('--valroot', required=True,
#        help='path to dataset')
    parser.add_argument('--workers', type=int, default=2,
        help='number of data loading workers')
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


def _read_image_as_array(path, dtype):
    f = Image.open(path).convert('L')
    try:
        image = np.asarray(f, dtype=dtype)
    finally:
        if hasattr(f, 'close'):
            f.close()
    return image


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

#        text = label
        text = self._lexicon[int(label)]
        return image, text, full_path

# loss = F.connectionist_temporal_classification(y_batch, t_batch, BLANK, x_length_batch, t_length_batch)


def main():
    args = arg()
    if not os.path.isdir(args.out):
        os.system('mkdir {0}'.format(args.out))


    nc = 1
    nclass = len(args.alphabet) + 1
    model = crnn.CRNN(args.imgH, nc, nclass, args.nh)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()


    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train = TextImageDataset(
        pairs_path='dataset/90kDICT32px/1ktrain.txt',
        lexicon=args.lexicon)

    for img in train[0:10]:
        print(img[0].shape, img[1], img[2])

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
        repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=args.gpu)
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
#    train_dataset = dataset.lmdbDataset(root=args.trainroot)
#    assert train_dataset
#    if not args.random_sample:
#        sampler = dataset.randomSequentialSampler(train_dataset, args.batchsize)
#    else:
#        sampler = None
#    train_loader = torch.utils.data.DataLoader(
#        train_dataset, batch_size=args.batchsize,
#        shuffle=True, sampler=sampler,
#        num_workers=int(args.workers),
#        collate_fn=dataset.alignCollate(imgH=args.imgH, imgW=args.imgW, keep_ratio=args.keep_ratio))
#    test_dataset = dataset.lmdbDataset(
#        root=args.valroot, transform=dataset.resizeNormalize((100, 32)))
#
#    nclass = len(args.alphabet) + 1
#    nc = 1
#
#    converter = utils.strLabelConverter(args.alphabet)
#    criterion = CTCLoss()
#
#
## custom weights initialization called on crnn
#    def weights_init(m):
#        classname = m.__class__.__name__
#        if classname.find('Conv') != -1:
#            m.weight.data.normal_(0.0, 0.02)
#        elif classname.find('BatchNorm') != -1:
#            m.weight.data.normal_(1.0, 0.02)
#            m.bias.data.fill_(0)
#
#
#    crnn = crnn.CRNN(args.imgH, nc, nclass, args.nh)
#    crnn.apply(weights_init)
#    if args.crnn != '':
#        print('loading pretrained model from %s' % args.crnn)
#        crnn.load_state_dict(torch.load(args.crnn))
#    print(crnn)
#
#    image = torch.FloatTensor(args.batchsize, 3, args.imgH, args.imgH)
#    text = torch.IntTensor(args.batchsize * 5)
#    length = torch.IntTensor(args.batchsize)
#
#    if args.cuda:
#        crnn.cuda()
#        crnn = torch.nn.DataParallel(crnn, device_ids=range(args.ngpu))
#        image = image.cuda()
#        criterion = criterion.cuda()
#
#    image = Variable(image)
#    text = Variable(text)
#    length = Variable(length)
#
## loss averager
#    loss_avg = utils.averager()
#
## setup optimizer
#    if args.adam:
#        optimizer = optim.Adam(crnn.parameters(), lr=args.lr,
#                               betas=(args.beta1, 0.999))
#    elif args.adadelta:
#        optimizer = optim.Adadelta(crnn.parameters(), lr=args.lr)
#    else:
#        optimizer = optim.RMSprop(crnn.parameters(), lr=args.lr)
#
#
#    def val(net, dataset, criterion, max_iter=100):
#        print('Start val')
#
#        for p in crnn.parameters():
#            p.requires_grad = False
#
#        net.eval()
#        data_loader = torch.utils.data.DataLoader(
#            dataset, shuffle=True, batch_size=args.batchsize, num_workers=int(args.workers))
#        val_iter = iter(data_loader)
#
#        i = 0
#        n_correct = 0
#        loss_avg = utils.averager()
#
#        max_iter = min(max_iter, len(data_loader))
#        for i in range(max_iter):
#            data = val_iter.next()
#            i += 1
#            cpu_images, cpu_texts = data
#            batch_size = cpu_images.size(0)
#            utils.loadData(image, cpu_images)
#            t, l = converter.encode(cpu_texts)
#            utils.loadData(text, t)
#            utils.loadData(length, l)
#
#            preds = crnn(image)
#            preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
#            cost = criterion(preds, text, preds_size, length) / batch_size
#            loss_avg.add(cost)
#
#            _, preds = preds.max(2)
#            preds = preds.squeeze(2)
#            preds = preds.transpose(1, 0).contiguous().view(-1)
#            sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
#            for pred, target in zip(sim_preds, cpu_texts):
#                if pred == target.lower():
#                    n_correct += 1
#
#        raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:args.n_test_disp]
#        for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts):
#            print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))
#
#        accuracy = n_correct / float(max_iter * args.batchsize)
#        print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))
#
#
#    def trainBatch(net, criterion, optimizer):
#        data = train_iter.next()
#        cpu_images, cpu_texts = data
#        batch_size = cpu_images.size(0)
#        utils.loadData(image, cpu_images)
#        t, l = converter.encode(cpu_texts)
#        utils.loadData(text, t)
#        utils.loadData(length, l)
#
#        preds = crnn(image)
#        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
#        cost = criterion(preds, text, preds_size, length) / batch_size
#        crnn.zero_grad()
#        cost.backward()
#        optimizer.step()
#        return cost
#
#
#    for epoch in range(args.epoch):
#        train_iter = iter(train_loader)
#        i = 0
#        while i < len(train_loader):
#            for p in crnn.parameters():
#                p.requires_grad = True
#            crnn.train()
#
#            cost = trainBatch(crnn, criterion, optimizer)
#            loss_avg.add(cost)
#            i += 1
#
#            if i % args.displayInterval == 0:
#                print('[%d/%d][%d/%d] Loss: %f' %
#                      (epoch, args.epoch, i, len(train_loader), loss_avg.val()))
#                loss_avg.reset()
#
#            if i % args.valInterval == 0:
#                val(crnn, test_dataset, criterion)
#
#            # do checkpointing
#            if i % args.saveInterval == 0:
#                torch.save(
#                    crnn.state_dict(), '{0}/netCRNN_{1}_{2}.pth'.format(args.experiment, epoch, i))
#
if __name__ == '__main__':
    main()
