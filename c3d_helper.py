# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 17:03:37 2016

@author: vision
"""

import lasagne
from lasagne.layers.shape import PadLayer
from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer
from lasagne.layers.dnn import Conv3DDNNLayer, MaxPool3DDNNLayer
from lasagne.nonlinearities import softmax
import pickle

def build_model():
    net = {}
    net['input'] = InputLayer((None, 3, 16, 112, 112))

    # ----------- 1st layer group ---------------
    net['conv1a'] = Conv3DDNNLayer(net['input'], 64, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify,flip_filters=False)
    net['pool1']  = MaxPool3DDNNLayer(net['conv1a'],pool_size=(1,2,2),stride=(1,2,2))

    # ------------- 2nd layer group --------------
    net['conv2a'] = Conv3DDNNLayer(net['pool1'], 128, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify)
    net['pool2']  = MaxPool3DDNNLayer(net['conv2a'],pool_size=(2,2,2),stride=(2,2,2))

    # ----------------- 3rd layer group --------------
    net['conv3a'] = Conv3DDNNLayer(net['pool2'], 256, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify)
    net['conv3b'] = Conv3DDNNLayer(net['conv3a'], 256, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify)
    net['pool3']  = MaxPool3DDNNLayer(net['conv3b'],pool_size=(2,2,2),stride=(2,2,2))

    # ----------------- 4th layer group --------------
    net['conv4a'] = Conv3DDNNLayer(net['pool3'], 512, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify)
    net['conv4b'] = Conv3DDNNLayer(net['conv4a'], 512, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify)
    net['pool4']  = MaxPool3DDNNLayer(net['conv4b'],pool_size=(2,2,2),stride=(2,2,2))

    # ----------------- 5th layer group --------------
    net['conv5a'] = Conv3DDNNLayer(net['pool4'], 512, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify)
    net['conv5b'] = Conv3DDNNLayer(net['conv5a'], 512, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify)
    # We need a padding layer, as C3D only pads on the right, which cannot be done with a theano pooling layer
    net['pad']    = PadLayer(net['conv5b'],width=[(0,1),(0,1)], batch_ndim=3)
    net['pool5']  = MaxPool3DDNNLayer(net['pad'],pool_size=(2,2,2),pad=(0,0,0),stride=(2,2,2))
    net['fc6-1']  = DenseLayer(net['pool5'], num_units=4096,nonlinearity=lasagne.nonlinearities.rectify)
    net['fc7-1']  = DenseLayer(net['fc6-1'], num_units=4096,nonlinearity=lasagne.nonlinearities.rectify)
    net['fc8-1']  = DenseLayer(net['fc7-1'], num_units=487, nonlinearity=None)
    net['prob']  = NonlinearityLayer(net['fc8-1'], softmax)

    return net

def set_weights(net,model_file):
    with open(model_file) as f:
        print('Load pretrained weights from %s...' % model_file)
        model = pickle.load(f)
    print('Set the weights...')
    lasagne.layers.set_all_param_values(net, model,trainable=True)