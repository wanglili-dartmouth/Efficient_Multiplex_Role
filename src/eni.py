import numpy as np
import math, os
import time
import shutil
import six
import os
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import network, utils
import numpy as np
np.set_printoptions(threshold=np.inf)
def attention(inputs, attention_size, time_major=False, return_alphas=False):
    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

    hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer

    initializer = tf.random_normal_initializer(stddev=0.1)

    # Trainable parameters
    w_omega = tf.get_variable(name="w_omega", shape=[hidden_size, attention_size], initializer=initializer)
    b_omega = tf.get_variable(name="b_omega", shape=[attention_size], initializer=initializer)
    u_omega = tf.get_variable(name="u_omega", shape=[attention_size], initializer=initializer)

    with tf.name_scope('v'):
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
    alphas = tf.nn.softmax(vu, name='alphas')         # (B,T) shape

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    if not return_alphas:
        return output
    else:
        return output, alphas
def squared_dist(A): 
    expanded_a = tf.expand_dims(A, 1)
    expanded_b = tf.expand_dims(A, 0)
    distances = tf.reduce_sum(tf.squared_difference(expanded_a, expanded_b), 2)
    return distances
class eni(object):
    def __init__(self, graph, args, sess):
        self.graph = graph
        self.args = args
        self.sess = sess
        self.degree_max = network.get_max_degree(self.graph)
        self.degree = network.get_degree(self.graph)
        self.save_path = os.path.join(self.args.save_path, '{}_{}_{}_{}'.format(self.args.save_suffix, self.args.embedding_size, self.args.alpha, self.args.lamb))
        
        self.build_model()
    
    def build_model(self):
        
        with tf.variable_scope('Placeholder'):
            self.nodes_placeholder = tf.placeholder(tf.int32, (self.args.batch_size, ), name='nodes_placeholder')
            self.seqlen1_placeholder = tf.placeholder(tf.int32, (self.args.batch_size,), name='seqlen1_placeholder')
            self.seqlen2_placeholder = tf.placeholder(tf.int32, (self.args.batch_size,), name='seqlen2_placeholder')
            self.seqlen3_placeholder = tf.placeholder(tf.int32, (self.args.batch_size,), name='seqlen3_placeholder')
            self.neighborhood1_placeholder = tf.placeholder(tf.int32, (self.args.batch_size, self.args.sampling_size), name='neighborhood1_placeholder')
            self.neighborhood2_placeholder = tf.placeholder(tf.int32, (self.args.batch_size, self.args.sampling_size), name='neighborhood2_placeholder')
            self.neighborhood3_placeholder = tf.placeholder(tf.int32, (self.args.batch_size, self.args.sampling_size), name='neighborhood3_placeholder')
            self.label1_placeholder = tf.placeholder(tf.float32, (self.args.batch_size,), name='label1_placeholder')
            self.label2_placeholder = tf.placeholder(tf.float32, (self.args.batch_size,), name='label2_placeholder')
            self.label3_placeholder = tf.placeholder(tf.float32, (self.args.batch_size,), name='label3_placeholder')
            
        self.data = network.next_batch(self.graph, self.degree_max, sampling=True, sampling_size=self.args.sampling_size)

        with tf.variable_scope('Embeddings'):
            self.embeddings = tf.get_variable('embeddings',
                    [len(self.graph), self.args.embedding_size],
                    initializer=tf.constant_initializer(utils.init_embedding(self.degree, self.degree_max, self.args.embedding_size)))

        with tf.variable_scope('ATT1'):
            cell1 = tf.contrib.rnn.DropoutWrapper(
                    tf.contrib.rnn.BasicRNNCell(num_units=self.args.embedding_size, activation=tf.tanh),
                    input_keep_prob=1.0, output_keep_prob=1.0)
            encoder_output1, state1 = tf.nn.bidirectional_dynamic_rnn(
                    cell1,
                    cell1,
                    tf.nn.embedding_lookup(self.embeddings, self.neighborhood1_placeholder),
                    dtype=tf.float32,
                    sequence_length=self.seqlen1_placeholder)
            decoder_cell1 = attention(encoder_output1,self.args.embedding_size)
            self.att1_output = decoder_cell1
             

        with tf.variable_scope('ATT2'):
            cell2 = tf.contrib.rnn.DropoutWrapper(
                    tf.contrib.rnn.BasicRNNCell(num_units=self.args.embedding_size, activation=tf.tanh),
                    input_keep_prob=1.0, output_keep_prob=1.0)
            encoder_output2, states2 = tf.nn.bidirectional_dynamic_rnn(
                    cell2,
                    cell2,
                    tf.nn.embedding_lookup(self.embeddings, self.neighborhood2_placeholder),
                    dtype=tf.float32,
                    sequence_length=self.seqlen2_placeholder)
            
            decoder_cell2 = attention(encoder_output2,self.args.embedding_size)
            self.att2_output =decoder_cell2


        with tf.variable_scope('ATT3'):
            cell3 = tf.contrib.rnn.DropoutWrapper(
                    tf.contrib.rnn.BasicRNNCell(num_units=self.args.embedding_size, activation=tf.tanh),
                    input_keep_prob=1.0, output_keep_prob=1.0)
            encoder_output3, states3 = tf.nn.bidirectional_dynamic_rnn(
                    cell3,
                    cell3,
                    tf.nn.embedding_lookup(self.embeddings, self.neighborhood3_placeholder),
                    dtype=tf.float32,
                    sequence_length=self.seqlen3_placeholder)
             
            decoder_cell3 = attention(encoder_output3,self.args.embedding_size)
            self.att3_output =decoder_cell3
            
            self.combine=tf.stack([self.att1_output,self.att2_output,self.att3_output],axis=1)

        with tf.variable_scope('Attention'):
            cell_att = tf.contrib.rnn.DropoutWrapper(
                    tf.contrib.rnn.BasicRNNCell(num_units=self.args.embedding_size, activation=tf.tanh),
                    input_keep_prob=1.0, output_keep_prob=1.0)
            encoder_output_att, states_att = tf.nn.bidirectional_dynamic_rnn(
                    cell_att,
                    cell_att,
                    self.combine,
                    dtype=tf.float32,
                    )
            
            decoder_cell_att = attention(encoder_output_att,self.args.embedding_size)
            self.att =decoder_cell_att
            


        with tf.variable_scope('Guilded'):
            #self.predict_info = tf.squeeze(tf.layers.dense(self.att, units=3, activation=utils.selu))
            self.predict_info = tf.squeeze(tf.layers.dense(tf.nn.embedding_lookup(self.embeddings, self.nodes_placeholder), units=3, activation=utils.selu))
            print("---")
            print(self.att2_output.shape)
            print(self.combine.shape)
            print(self.label1_placeholder.shape)
            print(tf.stack([self.label1_placeholder,self.label2_placeholder],axis=1).shape)
            print("---")
        with tf.variable_scope('Loss'):

            normalized_X = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.embeddings, self.nodes_placeholder), dim = 1)
            X_X = tf.matmul(normalized_X, normalized_X,    adjoint_b = True )
            normalized_Agg = tf.nn.l2_normalize(self.att, dim = 1)
            Agg_Agg = tf.matmul(normalized_Agg, normalized_Agg,    adjoint_b = True )
            self.structure_loss = tf.reduce_sum(tf.square(tf.subtract(X_X,Agg_Agg))) 
            self.print_op=tf.print('-')
            
            
            self.guilded_loss = tf.losses.mean_squared_error(self.predict_info, tf.stack([self.label1_placeholder,self.label2_placeholder,self.label3_placeholder],axis=1))
            self.total_loss = self.structure_loss+self.args.lamb*self.guilded_loss
        with tf.variable_scope('Optimizer'):
            self.optimizer = tf.train.RMSPropOptimizer(self.args.learning_rate)
            tvars = tf.trainable_variables()
            grads, self.global_norm = tf.clip_by_global_norm(tf.gradients(self.total_loss, tvars), self.args.grad_clip)
            self.train_op = self.optimizer.apply_gradients(zip(grads, tvars))

        with tf.variable_scope('Summary'):
            tf.summary.scalar("guilded_loss", self.guilded_loss)
            tf.summary.scalar("structure_loss", self.structure_loss)
            tf.summary.scalar("total_loss", self.total_loss)
            tf.summary.scalar("globol_norm", self.global_norm)
            for (grad, var) in zip(grads, tvars):
                if grad is not None:
                    tf.summary.histogram('grad/{}'.format(var.name), grad)
                    tf.summary.histogram('weight/{}'.format(var.name), var)

            log_dir = os.path.join(self.save_path, 'logs')
            if os.path.exists(log_dir):
                shutil.rmtree(log_dir)
            self.summary_writer = tf.summary.FileWriter(log_dir, self.sess.graph)

            config = projector.ProjectorConfig()
            embedding = config.embeddings.add()
            embedding.tensor_name = self.embeddings.name
            embedding.metadata_path = os.path.join(os.path.join(self.args.save_path, 'data', 'index.tsv'))
            projector.visualize_embeddings(self.summary_writer, config)

            self.merged_summary = tf.summary.merge_all()

        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())


    def fill_batch(self):
        inputs = [[] for _ in range(10)]
        for _ in range(self.args.batch_size):
            data, label1,label2,label3 = six.next(self.data)
            data += (label1, label2,label3)
            for input_, d in zip(inputs, data):
                input_.append(d)
        
        return {self.nodes_placeholder: inputs[0],
                self.neighborhood1_placeholder: inputs[1],
                self.seqlen1_placeholder: inputs[2],
                self.neighborhood2_placeholder: inputs[3],
                self.seqlen2_placeholder: inputs[4],
                self.neighborhood3_placeholder: inputs[5],
                self.seqlen3_placeholder: inputs[6],
                self.label1_placeholder: inputs[7],
                self.label2_placeholder: inputs[8],
                self.label3_placeholder: inputs[9]
                }

    def get_embeddings(self):
        return self.embeddings.eval(session=self.sess)[1:]

    # @profile
    def train(self):
        print('training')
        total_num = int((len(self.graph)-1)/self.args.batch_size)
        if(total_num<5):
            total_num=5
        print("batch_size: {}".format(self.args.batch_size))
        num = 0
        for epoch in range(self.args.epochs_to_train):
            guilded_loss = 0.0
            structure_loss = 0.0
            n = 0
            for i in range(total_num):
                begin = time.time()
                batch_data = self.fill_batch()
                #print(batch_data)
                _, total_loss, structure_loss,  guilded_loss,_ = self.sess.run([self.train_op, self.total_loss, self.structure_loss,self.guilded_loss,self.print_op], feed_dict=batch_data)
                n += 1
                end = time.time()
                #process = psutil.Process(os.getpid())
                print(("epoch: {}/{}, batch: {}/{}, loss: {:.6f}, structure_loss: {:.6f}, guilded_loss: {:.6f}, time: {:.4f}s").format(epoch, self.args.epochs_to_train, n-1, total_num, total_loss, structure_loss, guilded_loss, end-begin))
                if num % 5 == 0:
                    summary_str = self.sess.run(self.merged_summary, feed_dict=batch_data)
                    self.summary_writer.add_summary(summary_str, num)
                num += 1
            self.save_embeddings(self.save_path,epoch)

    def save_embeddings(self, save_path=None,epoch=0):
        print("Save embeddings in {}".format(save_path))
        embeddings = self.get_embeddings()
        filename = os.path.join(save_path, 'embeddings'+str(epoch)+'.npy')
        np.save(filename, embeddings)

    def save_model(self, step, name='eni'):
        save_path = self.save_path
        print("Save varibales in {}".format(save_path))
        self.saver.save(self.sess, os.path.join(save_path, 'eni'), global_step=step)

    def save(self, name='eni'):
        save_path = self.save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.save_embeddings(save_path)
        with open(os.path.join(save_path, 'config.txt'), 'w') as f:
            for key, value in vars(self.args).items():
                if value is None:
                    continue
                if type(value) == list:
                    s_v = " ".join(list(map(str, value)))
                else:
                    s_v = str(value)
                f.write(key+" "+s_v+'\n')

