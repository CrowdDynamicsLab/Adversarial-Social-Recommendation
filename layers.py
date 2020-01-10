import tensorflow as tf
import numpy as np
import flags
from flags import FLAGS
# Layer def

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).
    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off
    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])

def uniform(shape, scale=0.05, name=None):
    """Uniform init."""
    initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def ones(shape, name=None):
    """All ones."""
    initial = tf.ones(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


class CustomDense(Layer):
    """Dense layer."""


    def __init__(self, input_dim, output_dim, name = None, dropout=FLAGS.use_dropout, dropout_rate=FLAGS.dropout_rate, sparse_inputs=False, act=tf.nn.relu, bias=True, **kwargs):
        super(CustomDense, self).__init__(**kwargs)
        if name is None:
            name = self.name
        if dropout:
            self.dropout = dropout_rate
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.bias = bias

        with tf.variable_scope(name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs
        if self.sparse_inputs:
            x = tf.SparseTensor(indices=x.indices, 
                                values=tf.nn.dropout(x.values, 1.0 - self.dropout),
                                dense_shape=x.dense_shape)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # transform
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)




class residual_layer:

    def __init__(self, var_names, scaler_dims):

        self.layer = CustomDense(scaler_dims[0], scaler_dims[0], name = var_names[0], 
                                          dropout=FLAGS.use_dropout, dropout_rate=FLAGS.dropout_rate, 
                                          sparse_inputs=False)

        self.scaler = tf.Variable(tf.random.normal(
                            [scaler_dims[0],scaler_dims[0]],
                                mean=0.0,
                                stddev=0.01,
                                dtype=tf.float32,
                                seed=None,
                                name = var_names[1]))

        self.vars = {
                            "scaler": self.scaler
                        }
        self.vars.update(self.layer.vars)


    def __call__(self, input_vec):
        self.residual = tf.matmul(self.layer.__call__(input_vec), self.scaler)
        return self.residual



class residual_layer_hadamard_scaler:

    def __init__(self, var_names, scaler_dims):

        self.layer = CustomDense(scaler_dims[0], scaler_dims[0], name = var_names[0], 
                                          dropout=FLAGS.use_dropout, dropout_rate=FLAGS.dropout_rate, 
                                          sparse_inputs=False)

        self.scaler = tf.Variable(tf.random.normal(
                            [1,scaler_dims[0]],
                                mean=0.0,
                                stddev=1.0,
                                dtype=tf.float32,
                                seed=None,
                                name = var_names[1]))

        self.vars = {
                            "scaler": self.scaler
                        }
        self.vars.update(self.layer.vars)


    def __call__(self, input_vec):
        self.residual = tf.multiply(self.layer.__call__(input_vec), self.scaler)
        return self.residual




class multi_modal_stacked_attention:

    def __init__(self, core_dim, mode_dim, var_name):

        self.mode_layer_1 = CustomDense(mode_dim, core_dim, name = var_name + "_mode_layer_1", 
                                          dropout=FLAGS.use_dropout, dropout_rate=FLAGS.dropout_rate, 
                                          sparse_inputs=False)
        self.core_layer_hadamard = CustomDense(core_dim , core_dim, name = var_name + "_core_layer_hadamard", 
                                          dropout=FLAGS.use_dropout, dropout_rate=FLAGS.dropout_rate, 
                                          sparse_inputs=False)
        self.core_layer_residual = CustomDense(core_dim , core_dim, name = var_name + "_core_layer_residual", 
                                          dropout=FLAGS.use_dropout, dropout_rate=FLAGS.dropout_rate, 
                                          sparse_inputs=False)
        self.vars = {"1_weight": self.mode_layer_1.vars['weights'],
                        "1_bias": self.mode_layer_1.vars['bias'],
                        "2_weight": self.core_layer_hadamard.vars['weights'],
                        "2_bias": self.core_layer_hadamard.vars['bias'],
                        "3_weight": self.core_layer_residual.vars['weights'],
                        "3_bias": self.core_layer_residual.vars['bias']
                        }

    def __call__(self, core_tensor, mode_tensor):
        self.core_tensor = tf.add(tf.multiply(self.core_layer_hadamard.__call__(core_tensor), 
                                        self.mode_layer_1.__call__(mode_tensor)),
                                    self.core_layer_residual.__call__(core_tensor))
        return self.core_tensor

