import chainer
from chainer import Variable
import chainer.functions as F
import utils
import dataset
from PIL import Image

import models.crnn as crnn
import numpy as np


def main():
    img_path = './data/demo.png'
    alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
    gpu = -1

    model = crnn.CRNN(32, 1, 37, 256)
    if gpu >= 0:
        model = model.cuda()

#   model_path = './data/crnn.pth'
    #print('loading pretrained model from %s' % model_path)
    #model.load_state_dict(torch.load(model_path))

    converter = utils.strLabelConverter(alphabet)

    transformer = dataset.resizeNormalize((100, 32))
    image = Image.open(img_path).convert('L')
    image = transformer(image)
    print(image.shape)
    image = np.expand_dims(image, axis=0).astype(np.float32)
    print(image.shape)
    if gpu >= 0:
        image = cuda.to_gpu(image)
    image = Variable(image)

    with chainer.using_config('train', False):
        preds = model(image) # (26, 1, 37)

    preds = F.argmax(preds, axis=2) # (26, 1)
    preds = F.transpose(preds, axes=(1, 0)).reshape(-1) # (26,)

    preds_size = Variable(np.array([preds.shape[0]]))
    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    print('%-20s => %-20s' % (raw_pred, sim_pred))


if __name__ == '__main__':
    main()
