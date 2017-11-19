import numpy as np
import tensorflow as tf

from tensorflow.python.ops.nn import dynamic_rnn
from tensorflow.contrib.lookup.lookup_ops import MutableHashTable
from tensorflow.contrib.layers.python.layers import layers
from cell import GRUCell, BasicLSTMCell, BasicRNNCell

PAD_ID = 0
UNK_ID = 1
_START_VOCAB = ['_PAD', '_UNK']

class RNN(object):
    def __init__(self,
            num_symbols,
            num_embed_units,
            num_units,
            num_layers,
            num_labels,
            embed,
            learning_rate=0.5,
            max_gradient_norm=5.0,
	    keep_prob=1.,
	    weight_decay=1e-10,
            RNN_type="BasicRNN"):
        #todo: implement placeholders
        self.texts = tf.placeholder(dtype = tf.string, shape = [None, None])
        self.texts_length = tf.placeholder(dtype = tf.int32, shape = [None])
        self.labels = tf.placeholder(dtype = tf.int64, shape = [None])
        '''
        self.texts = tf.placeholder()  # shape: batch*len
        self.texts_length = tf.placeholder()  # shape: batch
        self.labels = tf.placeholder()  # shape: batch
        '''
        
        self.symbol2index = MutableHashTable(
                key_dtype=tf.string,
                value_dtype=tf.int64,
                default_value=UNK_ID,
                shared_name="in_table",
                name="in_table",
                checkpoint=True)
        # build the vocab table (string to index)
        # initialize the training process
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
	self.weight_decay = tf.Variable(float(weight_decay), trainable=False, dtype=tf.float32)
	self.keep_prob = tf.Variable(float(keep_prob), trainable=False, dtype=tf.float32)

        self.global_step = tf.Variable(0, trainable=False)
        self.epoch = tf.Variable(0, trainable=False)
        self.epoch_add_op = self.epoch.assign(self.epoch + 1)


        self.index_input = self.symbol2index.lookup(self.texts)   # batch*len
        
        # build the embedding table (index to vector)
        if embed is None:
            # initialize the embedding randomly
            self.embed = tf.get_variable('embed', [num_symbols, num_embed_units], tf.float32)
        else:
            # initialize the embedding by pre-trained word vectors
            self.embed = tf.get_variable('embed', dtype=tf.float32, initializer=embed)
        
        
        self.embed_input = tf.nn.embedding_lookup(self.embed, self.index_input) #batch*len*embed_unit

        
        if num_layers == 1:
            if RNN_type == "BasicRNN":
                cell = BasicRNNCell(num_units)
	        # cell = tf.contrib.rnn.BasicRNNCell(num_units)
            elif RNN_type == "GRU":
                cell = GRUCell(num_units)
            elif RNN_type == "LSTM":
                cell = BasicLSTMCell(num_units)
        

        outputs, states = dynamic_rnn(cell, self.embed_input, self.texts_length, dtype=tf.float32, scope="rnn")

        if RNN_type == "LSTM":
            self.y0 = states[1]
        else:
	    self.y0 = states

        self.y0_dp = tf.nn.dropout(self.y0, keep_prob = self.keep_prob)

	self.y1 = tf.layers.dense(inputs = self.y0_dp, units = 128, activation = tf.nn.sigmoid)
	self.y2 = tf.layers.dense(inputs = self.y1, units = num_labels)
	logits = self.y2

	'''
        self.W1 = tf.Variable(tf.truncated_normal(stddev = .1, shape = [num_units, 128]))
        self.b1 = tf.Variable(tf.constant(.1, shape = [128]))
        self.u1 = tf.matmul(self.y0_dp, self.W1) + self.b1
        self.y1 = tf.nn.sigmoid(self.u1)

        self.W2 = tf.Variable(tf.truncated_normal(stddev = .1, shape = [128, 5]))
        self.b2 = tf.Variable(tf.constant(.1, shape = [5]))
        self.u2 = tf.matmul(self.y1, self.W2) + self.b2
	'''

	# logits = tf.layers.dense(inputs = self.y1, units = 5)
	# logits = self.u2

        #todo: implement unfinished networks

	with tf.name_scope("l2_loss"):
		vars   = tf.trainable_variables() 
		self.lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in vars ]) * self.weight_decay

        self.loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=logits), name='loss') + self.lossL2
        mean_loss = self.loss / tf.cast(tf.shape(self.labels)[0], dtype=tf.float32)
        predict_labels = tf.argmax(logits, 1, 'predict_labels')
        self.accuracy = tf.reduce_sum(tf.cast(tf.equal(self.labels, predict_labels), tf.int32), name='accuracy')

        self.params = tf.trainable_variables()
            
        # calculate the gradient of parameters
	'''
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        gradients = tf.gradients(mean_loss, self.params)
        clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
        self.update = opt.apply_gradients(zip(clipped_gradients, self.params), global_step=self.global_step)
	'''
	self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step,var_list=self.params)

        tf.summary.scalar('loss/step', self.loss)
        for each in tf.trainable_variables():
            tf.summary.histogram(each.name, each)

        self.merged_summary_op = tf.summary.merge_all()
        
        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, 
                max_to_keep=3, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)

    def print_parameters(self):
        for item in self.params:
            print('%s: %s' % (item.name, item.get_shape()))
    
    def train_step(self, session, data, summary=False):
        input_feed = {self.texts: data['texts'],
                self.texts_length: data['texts_length'],
                self.labels: data['labels']}
        # output_feed = [self.loss, self.accuracy, self.gradient_norm, self.update]
        output_feed = [self.loss, self.accuracy, self.train_op]
        if summary:
            output_feed.append(self.merged_summary_op)
        return session.run(output_feed, input_feed)
