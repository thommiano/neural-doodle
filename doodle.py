#
# Copyright (c) 2016, Alex J. Champandard.
#

import os
import pickle
import argparse

import numpy as np
import scipy.optimize


parser = argparse.ArgumentParser(description='Generate a new image by applying style onto a content image.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--content',        required=True, type=str,        help='Content image path as optimization target.')
parser.add_argument('--content-weight', default=10.0, type=float,       help='Weight of content relative to style.')
parser.add_argument('--content-layers', default='4_2', type=str,        help='The layer with which to match content.')
parser.add_argument('--style',          required=True, type=str,        help='Style image path to extract patches.')
parser.add_argument('--style-weight',   default=25.0, type=float,       help='Weight of style relative to content.')
parser.add_argument('--style-layers',   default='3_1,4_1', type=str,    help='The layers to match style patches.')
parser.add_argument('--semantic-ext',   default='_sem.png', type=str,   help='File extension for the semantic maps.')
parser.add_argument('--semantic-weight', default=50.0, type=float,      help='Global weight of semantics vs. features.')
parser.add_argument('--output',         default='output.png', type=str, help='Output image path to save once done.')
parser.add_argument('--output-resolutions', default=3, type=int,        help='Number of image scales to process.')
parser.add_argument('--smoothness',     default=1E+1, type=float,       help='Weight of image smoothing scheme.')
parser.add_argument('--seed',           default='noise', type=str,      help='Seed image path, "noise" or "content".')
parser.add_argument('--iterations',     default=100, type=int,          help='Number of iterations to run each resolution.')
parser.add_argument('--device',         default='gpu0', type=str,       help='Index of the GPU number to use, for theano.')
parser.add_argument('--print-every',    default=10, type=int,           help='How often to log statistics to stdout.')
parser.add_argument('--save-every',     default=0, type=int,            help='How frequently to save PNG into `frames`.')
args = parser.parse_args()


os.environ.setdefault('THEANO_FLAGS', 'device=%s,floatX=float32,allow_gc=True,print_active_device=False' % (args.device))

import theano
import theano.tensor as T
import theano.tensor.nnet.neighbours

import lasagne
from lasagne.layers import Conv2DLayer as ConvLayer, Pool2DLayer as PoolLayer
from lasagne.layers import InputLayer, ConcatLayer


class Model(object):

    def __init__(self):
        self.pixel_mean = np.array([103.939, 116.779, 123.680], dtype=np.float32).reshape((3,1,1))

        self.setup_model()
        self.load_data()

    def setup_model(self):
        net = {}

        # First network for the main image.
        net['img']     = InputLayer((1, 3, None, None))
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
        net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, pad=1)
        net['main']    = net['conv4_2']

        # Second network for the semantic map.
        net['map'] = InputLayer((1, 3, None, None))
        net['map_2'] = PoolLayer(net['map'], 2, mode='average_exc_pad')
        net['map_3'] = PoolLayer(net['map'], 4, mode='average_exc_pad')
        net['map_4'] = PoolLayer(net['map'], 8, mode='average_exc_pad')

        net['sem2_1'] = ConcatLayer([net['conv3_1'], net['map_2']])
        net['sem3_1'] = ConcatLayer([net['conv3_1'], net['map_3']])
        net['sem4_1'] = ConcatLayer([net['conv4_1'], net['map_4']])

        # Third network for the nearest neighbors (default size for now).
        net['nn3_1'] = ConvLayer(net['sem3_1'], 1, 3, b=None, pad=0)
        net['nn4_1'] = ConvLayer(net['sem4_1'], 1, 3, b=None, pad=0)

        self.network = net

    def load_data(self): 
        data = pickle.load(open('vgg19_conv.pkl', 'rb'))
        params = lasagne.layers.get_all_param_values(self.network['main'])
        lasagne.layers.set_all_param_values(self.network['main'], data[:len(params)])

    def prepare(self, layers):
        self.tensor_img = T.tensor4()
        self.tensor_map = T.tensor4()
        self.tensor_inputs = {self.network['img']: self.tensor_img, self.network['map']: self.tensor_map}

        outputs = lasagne.layers.get_output([self.network[l] for l in layers], self.tensor_inputs)
        self.tensor_outputs = {k: v for k, v in zip(layers, outputs)}


class NeuralGenerator(object):

    def __init__(self):
        self.model = Model()
        self.iteration = 0

        self.model.prepare(layers=['sem3_1', 'sem4_1', 'conv4_2'])
        self.prepare_content()
        self.prepare_style()

        self.model.prepare(layers=['sem3_1', 'sem4_1', 'conv4_2', 'nn3_1', 'nn4_1'])
        self.prepare_optimization()

    def prepare_content(self):
        content_image = scipy.ndimage.imread(args.content, mode='RGB')
        self.content_image = self.prepare_image(content_image)
        self.content_map = np.zeros((1, 1)+self.content_image.shape[2:], dtype=np.float32)

    def prepare_style(self):
        style_image = scipy.ndimage.imread(args.style, mode='RGB')
        self.style_image = self.prepare_image(style_image)
        self.style_map = np.zeros((1, 1)+self.content_image.shape[2:], dtype=np.float32)

        for layer in args.style_layers.split(','):
            extractor = theano.function([self.model.tensor_img, self.model.tensor_map],
                                        self.extract_patches(self.model.tensor_outputs['sem'+layer]))
            patches, norm = extractor(self.style_image, self.style_map)

            l = self.model.network['nn'+layer]
            l.N = theano.shared(norm)
            l.W.set_value(patches)
            l.num_filters = patches.shape[0]

    def extract_patches(self, f, size=3, stride=1):
        patches = theano.tensor.nnet.neighbours.images2neibs(f, (size, size), (stride, stride), mode='valid')
        patches = patches.reshape((-1, patches.shape[0] // f.shape[1], size, size)).dimshuffle((1, 0, 2, 3))

        norm = T.sqrt(T.sum(patches ** 2.0, axis=(1,2,3), keepdims=True))
        return patches[:,:,::-1,::-1], norm

    def prepare_optimization(self):
        self.content_loss = []
        for layer in args.content_layers.split(','):
            content_features = self.model.tensor_outputs['conv'+layer].eval({self.model.tensor_img: self.content_image})
            content_loss = T.mean((self.model.tensor_outputs['conv'+layer] - content_features) ** 2.0)
            self.content_loss.append(args.content_weight * content_loss)

        def style_loss(l):
            layer = self.model.network['nn'+l]
            dist = self.model.tensor_outputs['nn'+l]
            patches, norm = self.extract_patches(self.model.tensor_outputs['sem'+l])
            dist = dist.reshape((dist.shape[1], -1)) / norm.reshape((1,-1)) / layer.N.reshape((-1,1))

            best = dist.argmax(axis=0)
            return T.mean((patches - layer.W[best]) ** 2.0)

        self.style_loss = [args.style_weight * style_loss(l) for l in args.style_layers.split(',')]

        variation_loss = args.smoothness * self.variation_loss(self.model.tensor_img)
        losses = self.content_loss + self.style_loss + [variation_loss]
        grad = T.grad(sum(losses), self.model.tensor_img)
        self.compute_grad_and_losses = theano.function([self.model.tensor_img, self.model.tensor_map],
                                                       [grad] + losses, on_unused_input='ignore')

    def variation_loss(self, x):
        return (((x[:,:,:-1,:-1] - x[:,:,1:,:-1])**2 + (x[:,:,:-1,:-1] - x[:,:,:-1,1:])**2)**1.25).mean()

    def evaluate(self, Xn):
        current_img = Xn.reshape(self.content_image.shape).astype(np.float32) - self.model.pixel_mean
        grads, *losses = self.compute_grad_and_losses(current_img, self.content_map)
        loss = sum(losses)

        scipy.misc.toimage(self.finalize_image(Xn), cmin=0, cmax=255).save('frames/test%04d.png'%self.iteration)

        print(self.iteration, 'losses', ' '.join(["{:8.3e}".format(l/1000) for l in losses]), 'gradients', np.min(grads), np.max(grads))

        self.iteration += 1
        return loss, np.array(grads).flatten().astype(np.float64)

    def run(self):
        # Xn = self.content_image[0] + self.model.pixel_mean
        Xn = np.random.uniform(64, 192, self.content_image.shape[2:] + (3,)).astype(np.float32)

        data_bounds = np.zeros((np.product(Xn.shape), 2), dtype=np.float64)
        data_bounds[:] = (0.0, 255.0)

        Xn, Vn, info = scipy.optimize.fmin_l_bfgs_b(
                            self.evaluate,
                            Xn.astype(np.float64).flatten(),
                            bounds=data_bounds,
                            factr=0.0, pgtol=0.0,            # Disable automatic termination by setting low threshold.
                            m=16,                            # Maximum correlations kept in memory by algorithm. 
                            maxfun=args.iterations-1,        # Limit number of calls to evaluate().
                            iprint=-1)                       # Handle our own logging of information.
                            
        scipy.misc.toimage(self.finalize_image(Xn), cmin=0, cmax=255).save(args.output)

    def prepare_image(self, image):
        image = np.swapaxes(np.swapaxes(image, 1, 2), 0, 1)[::-1, :, :]
        image = image.astype(np.float32) - self.model.pixel_mean
        return image[np.newaxis]

    def finalize_image(self, x):
        x = x.reshape(self.content_image.shape[1:])[::-1]
        x = np.swapaxes(np.swapaxes(x, 0, 1), 1, 2)
        return np.clip(x, 0, 255).astype('uint8')


if __name__ == "__main__":
    generator = NeuralGenerator()
    generator.run()
