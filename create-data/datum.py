# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 18:33:48 2016

@author: vision
"""

import numpy as np

LABEL_TYPE = np.uint32
LABEL_BYTES = 4
SHAPE_TYPE = np.uint32
SHAPE_BYTES = 4

class Datum4D():
    def __init__(self):
        self.value = None
        self.array = None
        self.label = None
        
    @staticmethod
    def array_to_datum(array,label):
        assert len(array.shape) == 4
        assert (array.dtype == np.int16)
        assert (isinstance(label,int))
        value = np.array([label],dtype=np.uint32).tostring()
        value += np.array(array.shape,dtype=np.uint32).tostring()
        value += array.tostring()
        
        d = Datum4D()
        d.value = value
        return d
        
    @staticmethod
    def fromstring(string):
        d = Datum4D()
        d.label = np.fromstring(string[0:4],dtype=np.uint32)[0]
        shape = np.fromstring(string[4:20],dtype=np.uint32)
        d.array = np.fromstring(string[20:],dtype=np.int16).reshape(shape)
        return d
