import numpy as np
import tensorflow as tf
from tensorflow.python.framework import constant_op
import sys
import json
import time
import random
import os
random.seed(1229)

from model import RNN, _START_VOCAB

tf.app.flags.DEFINE_boolean("is_train", True, "Set to False to inference.")
tf.app.flags.DEFINE_boolean("read_graph", False, "Set to False to build graph.")
tf.app.flags.DEFINE_integer("symbols", 18430, "vocabulary size.")
tf.app.flags.DEFINE_integer("labels", 5, "Number of labels.")
tf.app.flags.DEFINE_integer("epoch", 1, "Number of epoch.")
tf.app.flags.DEFINE_integer("embed_units", 300, "Size of word embedding.")
tf.app.flags.DEFINE_integer("units", 512, "Size of each model layer.")
tf.app.flags.DEFINE_integer("layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("batch_size", 16, "Batch size to use during training.")
tf.app.flags.DEFINE_string("data_dir", "./data", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./train", "Training directory.")
tf.app.flags.DEFINE_string("RNN_type", "LSTM", "Training directory.")
tf.app.flags.DEFINE_boolean("log_parameters", True, "Set to True to show the parameters")
tf.app.flags.DEFINE_float("learning_rate",   5e-5,  "Learning Rate")
tf.app.flags.DEFINE_float("keep_prob",   1., "dropout keep probability")
tf.app.flags.DEFINE_float("weight_decay",   1e-5, "L2 loss")

FLAGS = tf.app.flags.FLAGS

def load_data(path, fname):
    print('Creating %s dataset...' % fname)
    data = []
    with open('%s/%s' % (path, fname)) as f:
        for idx, line in enumerate(f):
            tokens = line.split(' ')
            data.append({'label':tokens[0], 'text':tokens[1:]})
    return data

def build_vocab(path, data):
    print("Creating vocabulary...")
    vocab = {}
    for i, pair in enumerate(data):
        for token in pair['text']:
            if token in vocab:
                vocab[token] += 1
            else:
                vocab[token] = 1
    vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)

    '''
    print len(vocab_list), " ", type(vocab_list)
    print vocab_list[0], ", ", vocab_list[100]
    '''

    if len(vocab_list) > FLAGS.symbols:
        vocab_list = vocab_list[:FLAGS.symbols]
    else:
        FLAGS.symbols = len(vocab_list)

    word2index = {}
    for i in range(len(vocab_list)):
        word2index[vocab_list[i]] = i

    print("Loading word vectors...")
    #todo: load word vector from 'vector.txt' to embed, where the value of each line is the word vector of the word in vocab_list
    '''
    embed = []
    embed = np.array(embed, dtype=np.float32)
    '''
    # calculated from vector.txt
    '''
    mean = -0.0055882638895467206
    std = 0.36930525876815262
    '''
    embed = np.random.randn(len(vocab_list), FLAGS.embed_units) * 1e-3
    # embed = np.zeros(shape = (len(vocab_list), FLAGS.embed_units))

    with open("%s/vector.txt" % (path)) as f:
        while True:
            l = f.readline()
            if l == None or len(l) == 0:
                break
            l = l.split(" ")
            index = word2index[l[0]]
            lv = len(vocab_list)
            for i in range(FLAGS.embed_units):
                embed[index][i] = float(l[i + 1])

    embed = embed.astype('float32')
    return vocab_list, embed

def gen_batch_data(data):
    def padding(sent, l):
        return sent + ['_PAD'] * (l-len(sent))

    max_len = max([len(item['text']) for item in data])
    texts, texts_length, labels = [], [], []
        
    for item in data:
        texts.append(padding(item['text'], max_len))
        texts_length.append(len(item['text']))
        labels.append(int(item['label']))

    batched_data = {'texts': np.array(texts), 'texts_length':texts_length, 'labels':labels}

    return batched_data

iteration = 0
tot_loss = 0.
tot_acc = 0.

def train(model, sess, dataset, summary_writer):
    global iteration, tot_loss, tot_acc
    st, ed, loss, accuracy = 0, 0, .0, .0
    gen_summary = True
    while ed < len(dataset):
        st, ed = ed, ed+FLAGS.batch_size if ed+FLAGS.batch_size < len(dataset) else len(dataset)
        batch_data = gen_batch_data(dataset[st:ed])
        outputs = model.train_step(sess, batch_data, summary=gen_summary)
        if gen_summary: 
            summary = outputs[-1]
            gen_summary = False
        loss += outputs[0]
        accuracy += outputs[1]

	iteration += 1
	tot_loss += outputs[0] / float(FLAGS.batch_size)
	tot_acc += outputs[1] / float(FLAGS.batch_size)

	freq = 20
	if iteration % 20 == 0:
		summary = tf.Summary()
		summary.value.add(tag='loss/train', simple_value=tot_loss / float(freq))
		summary.value.add(tag='accuracy/train', simple_value=tot_acc / float(freq))
		summary_writer.add_summary(summary, iteration)
		# print iteration, ": ", tot_loss / 30., " ", tot_acc / 30.
		tot_loss = 0.
		tot_acc = 0.
	
    sess.run(model.epoch_add_op)

    return loss / len(dataset), accuracy / len(dataset), summary

def evaluate(model, sess, dataset):
    st, ed, loss, accuracy = 0, 0, .0, .0
    while ed < len(dataset):
        st, ed = ed, ed+FLAGS.batch_size if ed+FLAGS.batch_size < len(dataset) else len(dataset)
        batch_data = gen_batch_data(dataset[st:ed])
        # outputs = sess.run(['loss:0', 'accuracy:0'], {'texts:0':batch_data['texts'], 'texts_length:0':batch_data['texts_length'], 'labels:0':batch_data['labels']})
        outputs = sess.run([model.loss, model.accuracy], {model.texts:batch_data['texts'], model.texts_length:batch_data['texts_length'], model.labels:batch_data['labels']})
        loss += outputs[0]
        accuracy += outputs[1]
    return loss / len(dataset), accuracy / len(dataset)

def inference(model, sess, dataset):
    st, ed, loss, accuracy = 0, 0, .0, .0
    result = []
    while ed < len(dataset):
        st, ed = ed, ed+FLAGS.batch_size if ed+FLAGS.batch_size < len(dataset) else len(dataset)
        batch_data = gen_batch_data(dataset[st:ed])
        outputs = sess.run(['predict_labels:0'], {'texts:0':batch_data['texts'], 'texts_length:0':batch_data['texts_length']})
        result += outputs[0].tolist()

    with open('result.txt', 'w') as f:
        for label in result:
            f.write('%d\n' % label)


'''
data_train = load_data(FLAGS.data_dir, 'train.txt')
data_dev = load_data(FLAGS.data_dir, 'dev.txt')
data_test = load_data(FLAGS.data_dir, 'test.txt')
vocab, embed = build_vocab(FLAGS.data_dir, data_train)
'''

# os.system("rm -r %s/log" % FLAGS.train_dir)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    if FLAGS.is_train:
        print(FLAGS.__flags)
        data_train = load_data(FLAGS.data_dir, 'train.txt')
        data_dev = load_data(FLAGS.data_dir, 'dev.txt')
        data_test = load_data(FLAGS.data_dir, 'test.txt')
        vocab, embed = build_vocab(FLAGS.data_dir, data_train)
        
	model = RNN(
		FLAGS.symbols, 
		FLAGS.embed_units,
		FLAGS.units, 
		FLAGS.layers,
		FLAGS.labels,
		embed,
		learning_rate = 0.,
		keep_prob = FLAGS.keep_prob,
		weight_decay = FLAGS.weight_decay,
                RNN_type = FLAGS.RNN_type)
	for lr in [1e-3]:
		for wd in [0.]:
			for kb in [1.]:
                                '''
				if not ((abs(lr - 3e-5) < 1e-10 and abs(wd - 3e-5) < 1e-10 and abs(kb - .5) < 1e-10) or (abs(lr - 6e-6) < 1e-10 and abs(wd - 1e-5) < 1e-10 and abs(kb - .7) < 1e-10)):
					continue
                                '''
				hyparam_str = "learning_rate_" + str(lr) + "__weight_decay_" + str(wd) + "___keep_prob_" + str(kb)
				with tf.variable_scope("test_lr"):
			        	if FLAGS.log_parameters:
			        	    model.print_parameters()
			        	
			        	# if tf.train.get_checkpoint_state(FLAGS.train_dir):
			        	if False:
			        	    print("Reading model parameters from %s" % FLAGS.train_dir)
			        	    model.saver.restore(sess, tf.train.latest_checkpoint(FLAGS.train_dir))
			        	else:
			        	    print("Created model with fresh parameters.")
			        	    tf.global_variables_initializer().run()
			        	    op_in = model.symbol2index.insert(constant_op.constant(vocab),
			        	        constant_op.constant(list(range(FLAGS.symbols)), dtype=tf.int64))
			        	    sess.run(op_in)
	
					sess.run(model.learning_rate.assign(lr))
					sess.run(model.weight_decay.assign(wd))
					sess.run(model.keep_prob.assign(kb))
	
					iteration = 0
					tot_loss = 0.
					tot_acc = 0.
			
			        	summary_writer = tf.summary.FileWriter(('%s/log/' + hyparam_str) % FLAGS.train_dir, sess.graph)
					#test_writer = tf.summary.FileWriter("%s/log/test" % FLAGS.train_dir)
			        	max_acc = 0.
			        	while model.epoch.eval() < FLAGS.epoch:
			        	    epoch = model.epoch.eval()
			        	    random.shuffle(data_train)
			        	    start_time = time.time()
			        	    loss, accuracy, summary = train(model, sess, data_train, summary_writer)

			        	    model.saver.save(sess, '%s/checkpoint' % FLAGS.train_dir, global_step=model.global_step)
			        	    print("epoch %d learning rate %.4f epoch-time %.4f loss %.8f accuracy [%.8f]" % (epoch, model.learning_rate.eval(), time.time()-start_time, loss, accuracy))
			        	    #todo: implement the tensorboard code recording the statistics of development and test set
			        	    loss, accuracy = evaluate(model, sess, data_dev)

			        	    summary = tf.Summary()
			        	    summary.value.add(tag='loss/dev', simple_value=loss)
			        	    summary.value.add(tag='accuracy/dev', simple_value=accuracy)
			        	    summary_writer.add_summary(summary, epoch)

			        	    print("        dev_set, loss %.8f, accuracy [%.8f]" % (loss, accuracy))
			
			        	    loss, accuracy = evaluate(model, sess, data_test)
			        	    summary = tf.Summary()
			        	    summary.value.add(tag='loss/test', simple_value=loss)
			        	    summary.value.add(tag='accuracy/test', simple_value=accuracy)
			        	    summary_writer.add_summary(summary, epoch)
			
			        	    if accuracy > max_acc:
			        	        max_acc = accuracy
			        	    print("        test_set, loss %.8f, accuracy [%.8f]" % (loss, accuracy))
			        	    print("        max test_accuracy [%.8f]" % (max_acc))
