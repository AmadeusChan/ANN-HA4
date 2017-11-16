import tensorflow as tf

class BasicRNNCell(tf.contrib.rnn.RNNCell):

    def __init__(self, num_units, activation=tf.tanh, reuse=None):
        self._num_units = num_units
        self._activation = activation
        self._reuse = reuse

        self.U = None
        self.W = None
        self.b = None

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "basic_rnn_cell", reuse=self._reuse):
            # pass
            #todo: implement the new_state calculation given inputs and state
            U = tf.get_variable(name = "U", shape =(state.shape[1], self._num_units), dtype = tf.float32, initializer = tf.truncated_normal_initializer(stddev = .1))
            W = tf.get_variable(name = "W", shape =(inputs.shape[1], self._num_units), dtype = tf.float32, initializer = tf.truncated_normal_initializer(stddev = .1))
            b = tf.get_variable(name = "b", shape = (self._num_units), dtype = tf.float32, initializer = tf.constant_initializer(.1))
            new_state = tf.matmul(state, U) + tf.matmul(inputs, W) + b

        return new_state, new_state

class GRUCell(tf.contrib.rnn.RNNCell):
    '''Gated Recurrent Unit cell (http://arxiv.org/abs/1406.1078).'''

    def __init__(self, num_units, activation=tf.tanh, reuse=None):
        self._num_units = num_units
        self._activation = activation
        self._reuse = reuse

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "gru_cell", reuse=self._reuse):
            pass
            #We start with bias of 1.0 to not reset and not update.
            #todo: implement the new_h calculation given inputs and state
            
        return new_h, new_h

class BasicLSTMCell(tf.contrib.rnn.RNNCell):
    '''Basic LSTM cell (http://arxiv.org/abs/1409.2329).'''

    def __init__(self, num_units, forget_bias=1.0, activation=tf.tanh, reuse=None):
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._activation = activation
        self._reuse = reuse

    @property
    def state_size(self):
        return (self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "basic_lstm_cell", reuse=self._reuse):
            c, h = state
            #For forget_gate, we add forget_bias of 1.0 to not forget in order to reduce the scale of forgetting in the beginning of the training.
            #todo: implement the new_c, new_h calculation given inputs and state (c, h)

            return new_h, (new_c, new_h)

def weight_variable(shape):
    initial = tf.truncated_normal(shape = shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)
