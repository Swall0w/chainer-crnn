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
                # input_size=512, hidden_size=nh, num_layers=nh
                L.NStepBiLSTM(n_layers=nh, in_size=512, out_size=nh, dropout=.0),
                # input_size=nh, hidden_size=nh, num_layers=nclass
                L.NStepBiLSTM(n_layers=nclass, in_size=nh, out_size=nh, dropout=.0),
            )

    def __call__(self, x):
        features = self.cnn(x)
        b, c, h, w = features.data.shape
        print('conv shape: ', features.data.shape)
        assert h ==1, 'the height of conv must be 1'
        features = F.squeeze(features, axis=2)
        features = F.transpose(features, axes=(2, 0, 1))
        output = self.rnn(features)
#        output = features
        return output


if __name__ == '__main__':
    import numpy as np
    from chainer import Variable
    #, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
    crnn =  CRNN(32, 1, 37, 256)
    img = np.random.randn(1, 32, 100).astype(np.float32)
    img = Variable(np.expand_dims(img, axis=0))
    print('image shape: ',img.data.shape)
    pred = crnn(img)
    print('pred shape: ', pred.data)
