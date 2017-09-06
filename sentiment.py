
"""
@author: mohitkumar
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import configparser

import data_utils
import lstm_model

tf.app.flags.DEFINE_float("learning_rate", 0.01, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 200,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("hidden_size", 50, "Size of each model layer.")
tf.app.flags.DEFINE_integer("max_epoch", 50, "no of epochs to run.")
tf.app.flags.DEFINE_integer("num_layers", 2, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("vocabulary_size", 40000, "English vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "/tmp", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "/tmp", "Training directory.")
tf.app.flags.DEFINE_integer("max_seq_size", 200,
                            "Limit on the size of training data.")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 50,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_boolean("use_fp16", False,
                            "Train using fp16 instead of fp32.")
tf.app.flags.DEFINE_float('dropout', 0.5, "dropout value")

FLAGS = tf.app.flags.FLAGS


def read_data(source_path):
    data_set=[]
    with tf.gfile.GFile(source_path, mode="r") as source_file:
            source= source_file.readline()
            #print("source", source)
            counter=0
            while source:
                counter+=1
                if counter % 10000 ==0:
                    #print("reading data line %d" %counter)
                    sys.stdout.flush()
                source_ids = [int(x) for x in source.split()]
                #print("source_ids", type(source_ids))
                data_set.append(source_ids)
                
                source = source_file.readline()
            
            return data_set
     

def train():
    train_ids, vocab = data_utils.prepare_data(
        FLAGS.data_dir,
        FLAGS.vocabulary_size,
        FLAGS.max_seq_size)
    
    with tf.Session() as sess:
        print("Creating %d layers of %d units." %(FLAGS.num_layers, FLAGS.hidden_size))
        model= create_model(sess)
        #print("model created")
        
        print("reading training data from the directory %s" % train_ids)
        data_set= read_data(train_ids)
        #print('training set loaded', train_set[:1])
        tr_set= np.vstack(data_set)
        #print("shape", tr_set.shape)
        
        np.random.shuffle(tr_set)
        num_batches= int(len(tr_set)/FLAGS.batch_size)
        
        # splitting data into train data and val data
        train_data= tr_set[0:int(0.8*len(tr_set))]
        test_data= tr_set[int(0.8 * len(tr_set))+1:len(tr_set)-1]
        
        #print("len of train_set", len(train_data))
        #print("len of test_set", len(test_data))
        
        
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        model.current_step_record(train=True)
        for step in xrange(num_batches*FLAGS.max_epoch):
            start_time= time.time()
            train_input, train_output, train_seq_length= model.get_batch(train_data)
            
            
            
            _, step_loss, _= model.step(sess, train_input, train_output, train_seq_length)
           
            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
            loss += step_loss / FLAGS.steps_per_checkpoint
            current_step += 1
            
            if current_step % FLAGS.steps_per_checkpoint == 0:
                perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                print ("global step %d learning rate %.4f step-time %.2f perplexity "
               "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                         step_time, perplexity))
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                
                previous_losses.append(loss)
                print("saving checkpoint")
                checkpoint_path = os.path.join(FLAGS.train_dir, "sentiment.ckpt")
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                
                step_time, val_loss = 0.0, 0.0
                
                validation_accuracy=0.0
                
                #print("running validation set")
                model.current_step_record(test= True)
                #tst_batch_no=0
                for val_step in xrange(int(len(test_data)/FLAGS.batch_size)):
                    test_inputs, test_targets, test_seq_length= model.get_batch(train_data, test_data= test_data)
                    test_loss, _, test_accuracy= model.step(sess, test_inputs, test_targets, test_seq_length, True)
                    val_loss+=test_loss
                    validation_accuracy+=test_accuracy
                    #print("test accuracy", test_accuracy)
                    
                print("Avg Test Loss: ", val_loss/len(test_data))
                print("accuracy: ", validation_accuracy/len(test_data)) 
                sys.stdout.flush()
            
  
  
def create_model(session):
    dtype= tf.int32
    model=lstm_model.SentimentModel(FLAGS.vocabulary_size, 
                                   FLAGS.hidden_size, 
                                   FLAGS.num_layers, 
                                   FLAGS.dropout, 
                                   FLAGS.max_gradient_norm, 
                                   FLAGS.learning_rate, 
                                   FLAGS.max_seq_size, 
                                   FLAGS.batch_size, 
                                   FLAGS.learning_rate_decay_factor)
    
    ckpt= tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("creating a fresh model")
        session.run(tf.global_variables_initializer())
    return model
        
    
    
    
    
def main(_):
    train()
        
        
if __name__== "__main__":
    tf.app.run()
