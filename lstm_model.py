
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from six.moves import xrange


class SentimentModel(object):
    
    
    def __init__(self, vocabulary_size, hidden_size, num_layers, dropout, gradient_clip, 
                 learning_rate, max_sequence_length, batch_size, lr_decay_rate, forward_only= False):
        
        self.vocabulary_size = vocabulary_size
        self.num_layers= num_layers
        self.dropout= dropout
        self.gradient_clip= gradient_clip
        self.learning_rate= tf.Variable(float(learning_rate), trainable= False, dtype= tf.float32)
        self.max_sequence_length= max_sequence_length
        self.batch_size= batch_size
        
        
        self.global_step= tf.Variable(0, trainable= False)
        self.learning_rate_decay_op= self.learning_rate.assign(self.learning_rate * lr_decay_rate)
        
        self.num_classes= 2
        self.input= tf.placeholder(tf.int32, shape=[None, max_sequence_length], name= 'input')
        self.output= tf.placeholder(tf.int32, shape=[None, self.num_classes], name='output')
        self.input_length= tf.placeholder(tf.int32, shape=[None], name='end_of_sequence')
        
        
     #   def loss(labels, logits):
      #      local_inputs= tf.cast(logits, tf.float32)
       #     
        #    losses=tf.cast(tf.nn.softmax_cross_entropy_with_logits(labels= labels, logits= local_inputs), tf.float32)
         #   sum_loss= tf.reduce_sum(losses)
          #  mean_loss= tf.reduce_mean(losses)
            
           # return losses, sum_loss, mean_loss
        
        
        
        self.dropout_keep_embedd_prob= tf.placeholder(tf.float32, name= "dropout_embedding_probability")
        self.dropout_keep_input_prob= tf.placeholder(tf.float32, name= 'dropout_input_probability')
        self.dropout_keep_output_prob= tf.placeholder(tf.float32, name="dropout_output_probability")
        
       # def embedding(hidden_size):
        #        
         #       w= tf.get_variable("W_embed", shape= [self.vocabulary_size, hidden_size],
          #                         initializer= tf.random_uniform_initializer(-1.0, 1.0))
           #     embedded_tokens= tf.nn.embedding_lookup(w, self.input)
            #    embedded_tokens_drop= tf.nn.dropout(embedded_tokens, keep_prob=self.dropout_keep_embedd_prob)
             #   return embedded_tokens_drop
       # with tf.variable_scope("rnn_input", reuse= True):
        #    rnn_input= [embedding(hidden_size)[:, i, :] for i in range(self.max_sequence_length)]
        with tf.variable_scope("embedding"), tf.device("/cpu:0"):
            W = tf.get_variable("W",
				[self.vocabulary_size, hidden_size],
				initializer=tf.random_uniform_initializer(-1.0, 1.0))
            embedded_tokens = tf.nn.embedding_lookup(W, self.input)
            embedded_tokens_drop = tf.nn.dropout(embedded_tokens, self.dropout_keep_embedd_prob)
        
        rnn_input = [embedded_tokens_drop[:, i, :] for i in range(self.max_sequence_length)]
            
        def single_cell():
            return rnn.DropoutWrapper(rnn.LSTMCell(hidden_size, initializer= tf.random_uniform_initializer(-1.0, 1.0),
                                state_is_tuple= True), input_keep_prob=self.dropout_keep_input_prob ,
                                    output_keep_prob=self.dropout_keep_output_prob )
            
        cell= single_cell()
        
        if num_layers>1:
            cell = rnn.MultiRNNCell([single_cell() for _ in range(num_layers)], state_is_tuple= True)
            
            initial_state = cell.zero_state(self.batch_size, tf.float32)
            
            rnn_output, rnn_state= rnn.static_rnn(cell= cell, inputs=rnn_input , 
                                                  initial_state= initial_state, 
                                                  sequence_length= self.input_length)
            
        
        w_t= tf.get_variable("w_layer", [hidden_size, self.num_classes], 
                             initializer= tf.truncated_normal_initializer(stddev=0.1))
        b_t= tf.get_variable("b_layer", [self.num_classes], initializer= tf.constant_initializer(0.1))
        
        self.scores= tf.nn.xw_plus_b(rnn_state[-1][0], w_t, b_t)
        self.y= tf.nn.softmax(self.scores)
        self.prediction= tf.argmax(self.scores, 1)
        
        self.losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores,
																  labels=self.output,
																  name="ce_losses")
        self.sum_loss = tf.reduce_sum(self.losses)
        self.mean_loss = tf.reduce_mean(self.losses)
        #self.losses, sum_loss, mean_loss= loss(self.scores, self.output)
        
        #self.accuracy=tf.reduce_mean(tf.cast(tf.equal(self.prediction, tf.argmax(self.output)), tf.float32), name= "accuracy")
        self.correct_predictions = tf.equal(self.prediction, tf.argmax(self.output, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")
        
        params= tf.trainable_variables()
        
        if not forward_only:
            self.gradient_norms = []
            self.updates = []
            opt= tf.train.AdamOptimizer(self.learning_rate)
            gradients= tf.gradients(self.losses, params)
            clipped_gradients, norm = tf.clip_by_global_norm(gradients, self.gradient_clip)
            self.gradient_norms.append(norm)
            self.updates.append(opt.apply_gradients(zip(clipped_gradients, params), 
                                                    global_step=self.global_step))
            
        self.saver= tf.train.Saver(tf.global_variables())
        
        
        
    def step(self, session, inputs, outputs, input_length, forward_only= False):
        #print("running a sentiment step")
        #print("input shape", inputs.shape)
        #print("output_shape", outputs.shape)
        #print("input_length", input_length.shape)
        input_feed={}
        
        input_feed[self.input.name]= inputs
        input_feed[self.output.name]= outputs
        input_feed[self.input_length.name]= input_length
        
        input_feed[self.dropout_keep_embedd_prob.name]= self.dropout
        input_feed[self.dropout_keep_input_prob.name]=self.dropout
        input_feed[self.dropout_keep_output_prob.name]=self.dropout
        #print("input feed", input_feed)
        if not forward_only:
            output_feed= [self.updates,  # Update Op that does SGD.
                     self.gradient_norms,  # Gradient norm.
                     self.mean_loss]
            
        else:
            output_feed= [self.mean_loss, self.y, self.accuracy]
            
        outputs= session.run(output_feed, input_feed)
        #print("session run finished in step")
        if not forward_only:
            return outputs[1], outputs[2], None
        else:
            return outputs[0], outputs[1], outputs[2]
   
    
    
    def current_step_record(self, train=False, test=False):
        if train:
            self.train_batch_no=0
        if test:
            self.test_batch_no=0
        
    def get_batch(self, train_data, test_data= None):
        #print("running get_batch")
        num_classes=2
        
            
        if not test_data:
        
            #tr_batch_no=current_step_record()
            #print("train_data_size", train_data.shape)
            
            #print("train_transpose", train_data.transpose().shape)
            train_targets = (train_data.transpose()[-1]).transpose()
            train_ohe= np.eye(num_classes)[train_targets]
            #print("target inputs", train_targets)
            #print("shape", train_targets.shape)
            
            train_input_size= (train_data.transpose()[-2]).transpose()
            train_input= (train_data.transpose()[0:-2]).transpose()
            
            train_num_batches= len(train_input)/self.batch_size
            #if self.train_batch_no==0:
                #print("train_batch_no", self.train_batch_no+1)
            #else:
                #print("train batch_no", (self.train_batch_no/200)+1)
            input_batch= train_input[self.train_batch_no:self.train_batch_no+200]
            output_batch= train_ohe[self.train_batch_no: self.train_batch_no+200]
            input_size_batch= train_input_size[self.train_batch_no:self.train_batch_no+200]
            self.train_batch_no+=200
            self.train_batch_no = self.train_batch_no % len(train_data)
            
            #print("train_batch_size", input_batch.shape)
            #print("train output size", output_batch.shape)
            return input_batch, output_batch, input_size_batch
        
        else:
            #print("tst batch_no", tst_batch_no)
            
            #print("test_data", test_data.shape)
            test_targets= (test_data.transpose()[-1]).transpose()
            test_ohe= np.eye(num_classes)[test_targets]
            
            test_input_size= (test_data.transpose()[-2]).transpose()
            test_input= (train_data.transpose()[0:-2]).transpose()
            test_num_batches= len(test_input)/self.batch_size
            
            test_input= test_input[:len(test_input)-(len(test_input)%self.batch_size)]
            #if self.test_batch_no==0:
                #print("test batch_no", self.test_batch_no+1)
            #else:
                #print("test batch_no", (self.test_batch_no/200)+1)
            test_input_batch= test_input[self.test_batch_no:self.test_batch_no+200]
            test_output_batch= test_ohe[self.test_batch_no: self.test_batch_no+200]
            test_input_size_batch= test_input_size[self.test_batch_no:self.test_batch_no+200]
            self.test_batch_no+=200
            #self.test_batch_no = self.test_batch_no % len(self.test_data)
            
            #print("test_input_size",test_input_batch.shape )
            
            return test_input_batch, test_output_batch, test_input_size_batch
            
          
        
       
       
            
    
        
            
            
            
        
        
            
            
            
        
            
        
            
            
            
        
        
        
        
        
        
        