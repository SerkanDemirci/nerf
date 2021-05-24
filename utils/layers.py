# Utility layer definitions
import numpy as np
import tensorflow as tf
from .interpolate import trilinear_interpolation

class GridEncoding3D(tf.keras.layers.Layer):
    """3d Grid Layer
    """
    def __init__(self, nbins : int, out_dims : int, bounds=None, transform=False, interp=True):
        """Initializes the 3D Grid layer

        Args:
            nbins (int): Number of cells per dimension (total number of cells=nbins ** 3) 
            out_dims (int): Number of the scalars stored inside each cell
            bounds (numpy.ndarray, optional): Bounds of the input
        """
        
        super().__init__()

        self.nbins = nbins
        self.bounds = bounds
        self.out_dims = out_dims
        self.transform = transform
        self.interp = interp

    def build(self, input_shape):
        self.grid = self.add_weight(shape=(self.nbins, self.nbins, self.nbins, self.out_dims))
        self.tr = tf.keras.layers.Dense(3, activation=None)

    def call(self, inputs):
        if self.bounds is not None:
            minval = self.bounds[0, :][None, :]
            step   = (self.bounds[1, :] - self.bounds[0, :])[None, :]
            inputs = (inputs - minval) / step

        if self.transform:
            inputs = self.tr(inputs)

        inputs = tf.clip_by_value(inputs * self.nbins, 0, self.nbins - 1)

        if self.interp:
            vals = trilinear_interpolation(self.grid, inputs)
        else:
            inputs = tf.cast(tf.floor(inputs), tf.int32)
            
            vals = 5 * tf.gather_nd(params=self.grid, indices=inputs, batch_dims=0)

        return vals

# class FourierLayer(tf.keras.layers.Layer):
#     '''
#     '''
#     def __init__(self, out_dims, ):
#         super().__init__()
#         self.out_dims = out_dims
 
#     def build(self, input_shape):
#         self.dense1 = tf.keras.layers.Dense()
#     def call(self, inputs):

