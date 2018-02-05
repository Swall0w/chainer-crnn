import chainer
import chainer.links as L
import chainer.functions as F

from collections import OrderedDict


class Sequential(chainer.Chain):
    """A part of the code has been borrowed from https://github.com/musyoku/chainer-sequential-chain
    """
    def __init__(self, *args):
        super(Sequential, self).__init__()
        assert len(args) > 0
        assert not hasattr(self, "layers")
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            self.layers = args[0].values()
            with self.init_scope():
                for key, layer in args[0].items():
                    if isinstance(layer, (chainer.Link, chainer.Chain, chainer.ChainList)):
                        setattr(self, key, layer)
        else:
            self.layers = args
            with self.init_scope():
                for idx, layer in enumerate(args):
                    if isinstance(layer, (chainer.Link, chainer.Chain, chainer.ChainList)):
                        setattr(self, str(idx), layer)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
