import chainer
import chainer.functions as F
import chainer.links as L

from sequential import Sequential
from collections import OrderedDict


class MaxPooling2D(object):
    def __init__(self, ksize, stride=None, pad=0, cover_all=True):
        self.args = [ksize, stride, pad, cover_all]

    def __call__(self, x):
        return F.max_pooling_2d(x, *self.args)


class LeakyRelu(object):
    def __init__(self, slope=0.2):
        self.args = [slope]

    def __call__(self, x):
        return F.leaky_relu(x, *self.args)


class BidirectionalLSTM(chainer.Chain):
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        self.bi_lstm = L.NStepBiLSTM(n_layers=1, in_size=nIn, out_size=nHidden, dropout=.0)
        self.embedding = L.Linear(nHidden*2, nOut)

    def __call__(self, x):
        _, _, recurrent = self.bi_lstm(hx=None, cx=None, xs=[item for item in x])
        recurrent = F.stack(recurrent, axis=0)
        T, b, h = recurrent.data.shape
        t_rec = F.reshape(recurrent, (T * b, h))
        output = self.embedding(t_rec)
        output = F.reshape(output, (T, b, -1))
        return output


class CRNN(chainer.Chain):
    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = OrderedDict()
        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i -1]
            nOut = nm[i]
            cnn['conv{0}'.format(i)] = L.Convolution2D(
                nIn, nOut, ksize=ks[i], stride=ss[i], pad=ps[i])

            if batchNormalization:
                cnn['bn{0}'.format(i)] = L.BatchNormalization(nOut)
            if leakyRelu:
                cnn['relu{0}'.format(i)] = LeakyRelu
            else:
                cnn['relu{0}'.format(i)] = F.relu

        convRelu(0)
        #cnn['pooling{0}'.format(0)] = MaxPooling2D(ksize=2, stride=2, pad=1)
        cnn['pooling{0}'.format(0)] = MaxPooling2D(2, 2)
        convRelu(1)
        cnn['pooling{0}'.format(1)] = MaxPooling2D(2, 2)
        convRelu(2, True)
        convRelu(3)
        cnn['pooling{0}'.format(2)] = MaxPooling2D(
            ksize=(2, 2), stride=(2, 1), pad=(0, 1))
        convRelu(4, True)
        convRelu(5)
        cnn['pooling{0}'.format(3)] = MaxPooling2D(
            ksize=(2, 2), stride=(2, 1), pad=(0, 1))
        convRelu(6, True)

        with self.init_scope():
            self.cnn = Sequential(cnn)
            self.rnn = Sequential(
                BidirectionalLSTM(512, nh, nh),
                BidirectionalLSTM(nh, nh, nclass)
            )

    def __call__(self, x):
        features = self.cnn(x)
        b, c, h, w = features.data.shape
        assert h ==1, 'the height of conv must be 1'
        features = F.squeeze(features, axis=2)
        features = F.transpose(features, axes=(2, 0, 1))
        output = self.rnn(features)
        return output


if __name__ == '__main__':
    import numpy as np
    from chainer import Variable
    # imgH, nc, nclass, nh
    crnn =  CRNN(32, 1, 37, 256)
    img = np.random.randn(1, 32, 100).astype(np.float32)
    img = Variable(np.expand_dims(img, axis=0))
    print('image shape: ',img.data.shape)
    pred = crnn(img)
    print('pred shape: ', pred.data.shape)
