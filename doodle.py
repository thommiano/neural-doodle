import pickle

import numpy as np
import scipy.optimize
import theano.tensor as T

import lasagne
from lasagne.layers import Conv2DLayer as ConvLayer, Pool2DLayer as PoolLayer
from lasagne.layers import InputLayer, ConcatLayer


class Model(object):

    def __init__(self, layers):
        self.layers = layers

        self.build_model()
        self.load_params()

    def build_model(self):
        net = {}

        # Main network for the primary image.        
        net['img']   = InputLayer((1, 3, None, None))
        net['conv1_1'] = ConvLayer(net['img'],   64, 3, pad=1)
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
        net['main']    = ConvLayer(net['conv4_1'], 512, 3, pad=1)

        # Secondary network for the semantic map.
        net['map']   = InputLayer((1, 3, None, None))
        net['map_3'] = PoolLayer(net['map'], 4, mode='average_exc_pad')
        net['map_4'] = PoolLayer(net['map'], 8, mode='average_exc_pad')

        net['sem3_1'] = ConcatLayer([net['conv3_1'], net['map_3']])
        net['sem4_1'] = ConcatLayer([net['conv4_1'], net['map_4']])

        self.network = net

    def load_params(self): 
        vgg19_values = pickle.load(open('vgg19_conv.pkl', 'rb'))
        params = lasagne.layers.get_all_param_values(self.network['main'])
        lasagne.layers.set_all_param_values(self.network['main'], vgg19_values[:len(params)])

        self.tensor_img = T.tensor4()
        self.tensor_map = T.tensor4()
        self.tensor_inputs = {self.network['img']: self.tensor_img, self.network['map']: self.tensor_map}
        self.tensor_outputs = lasagne.layers.get_output([self.network[l] for l in self.layers], self.tensor_inputs)


class NeuralGenerator(object):

    def __init__(self):
        self.model = Model(layers=['sem3_1'])

    def evaluate(self, Xn):
        return 0.0, np.zeros_like(Xn)

    def run(self):
        # img = np.zeros((1, 3, 512, 512), dtype=np.float32)
        # map = np.zeros((1, 1, 512, 512), dtype=np.float32)

        Xn = np.zeros((1, 3, 512, 512), dtype=np.float32)
        data_bounds = np.zeros((np.product(Xn.shape), 2), dtype=np.float64)
        data_bounds[:] = (0.0, 255.0)

        Xn, Vn, info = scipy.optimize.fmin_l_bfgs_b(
                            self.evaluate,
                            Xn.astype(np.float64).flatten(),
                            bounds=data_bounds,
                            factr=0.0, pgtol=0.0,            # Disable automatic termination by setting low threshold.
                            m=16,                            # Maximum correlations kept in memory by algorithm. 
                            maxfun=100,                      # Limit number of calls to evaluate().
                            iprint=-1)                       # Handle our own logging of information.


if __name__ == "__main__":
    generator = NeuralGenerator()
    generator.run()