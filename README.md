# Convolutional Recurrent Neural Network
This repository implements [the Convolutional Recurrent Neural Network(CRNN)](https://arxiv.org/abs/1507.05717) in chainer.

You can find [original implementation](https://github.com/bgshih/crnn)
and [Pytorch implementation](https://github.com/meijieru/crnn.pytorch).

# Demo
The demo script for CRNN can be found in demo.py.
Before running the demo, download a pretrained model from here.
This pretrained model is trained using train script.

The demo without any augments reads an example image and recognizes its text content.

Put the downloaded model file crnn.weights into directory data/.
Then launch the demo by:
```
    python3 demo.py
    a-----v--a-i-l-a-bb-l-ee-- => available
```

## Prerequisites
* Python 3.6.1 :: Anaconda custom (64-bit)
* chainer >= 3.00
* cupy
* numpy

## Train
I will write how to setup dataset and train model later.

Dataset for training is [Synthetic Word Dataset](http://www.robots.ox.ac.uk/~vgg/data/text/)

# Author
* Swall0w

# License
* MIT License
