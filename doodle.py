import pickle
import collections

import theano.tensor as T

import lasagne
from lasagne.layers import Conv2DLayer as ConvLayer, Pool2DLayer as PoolLayer
from lasagne.layers import InputLayer


class Model(object):

    def __init__(self, layers):
        self.layers = layers

        self.build_model()
        self.load_params()

    def build_model(self):
        net = collections.OrderedDict()
        net['input']   = InputLayer((1, 3, None, None))
        net['conv1_1'] = ConvLayer(net['input'],   64, 3, pad=1)
        net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3, pad=1)
        net['pool1']   = PoolLayer(net['conv1_2'], 2, mode='average_exc_pad')
        net['conv2_1'] = ConvLayer(net['pool1'],   128, 3, pad=1)
        net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3, pad=1)
        net['pool2']   = PoolLayer(net['conv2_2'], 2, mode='average_exc_pad')
        net['conv3_1'] = ConvLayer(net['pool2'],   256, 3, pad=1)
        net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3, pad=1)
        net['conv3_3'] = ConvLayer(net['conv3_2'], 256, 3, pad=1)
        net['conv3_4'] = ConvLayer(net['conv3_3'], 256, 3, pad=1)
        net['pool3']   = PoolLayer(net['conv3_4'], 2, mode='average_exc_pad')
        net['conv4_1'] = ConvLayer(net['pool3'],   512, 3, pad=1)
        net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, pad=1)
        # net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3, pad=1)
        # net['conv4_4'] = ConvLayer(net['conv4_3'], 512, 3, pad=1)
        # net['pool4']   = PoolLayer(net['conv4_4'], 2, mode='average_exc_pad')
        # net['conv5_1'] = ConvLayer(net['pool4'],   512, 3, pad=1)
        # net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3, pad=1)
        # net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3, pad=1)
        # net['conv5_4'] = ConvLayer(net['conv5_3'], 512, 3, pad=1)
        # net['pool5']   = PoolLayer(net['conv5_4'], 2, mode='average_exc_pad')
        self.network = net

    def load_params(self): 
        vgg19_values = pickle.load(open('vgg19_conv.pkl', 'rb'))
        output_layer = list(self.network.values())[-1]
        params = lasagne.layers.get_all_param_values(output_layer)
        lasagne.layers.set_all_param_values(output_layer, vgg19_values[:len(params)])

        self.tensor_input = T.tensor4()
        self.tensor_outputs = lasagne.layers.get_output([self.network[l] for l in self.layers], self.tensor_input)


if __name__ == "__main__":
    model = Model(layers=['conv3_1', 'conv4_1'])
