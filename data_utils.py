#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mohitkumar
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import glob
from six.moves import urllib

import numpy as np
from tensorflow.python.platform import gfile
import tensorflow as tf


_PAD = b"_PAD"
_UNK = b"_UNK"

PAD_ID = 0
UNK_ID = 1


_START_VOCAB = [_PAD, _UNK]
_WORD_SPLIT = re.compile(b"([!?\"/:;)(])")

def maybe_download(directory, filename, url):
  """Download filename from url unless it's already in directory."""
  if not os.path.exists(directory):
    print("Creating directory %s" % directory)
    os.mkdir(directory)
  filepath = os.path.join(directory, filename)
  if not os.path.exists(filepath):
    print("Downloading %s to %s" % (url, filepath))
    filepath, _ = urllib.request.urlretrieve(url, filepath)
    statinfo = os.stat(filepath)
    print("Successfully downloaded", filename, statinfo.st_size, "bytes")
  return filepath

def listdir_nohidden(path):
    return glob.glob(os.path.join(path, '*'))

def gunzip_file(gz_path, new_path):
  """Unzips from gz_path into new_path."""
  print("Unpacking %s to %s" % (gz_path, new_path))
  with gzip.open(gz_path, "rb") as gz_file:
    with open(new_path, "wb") as new_file:
      for line in gz_file:
        new_file.write(line)
        
        
def basic_tokenizer(sentence):
    words=[]
    sent= re.sub(b"[^\w']",b' ', sentence)
    for space_separated_fragment in sent.strip().split():
        words.extend(_WORD_SPLIT.split(space_separated_fragment))
    return [w for w in words if w]


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size):

    
    if not gfile.Exists(vocabulary_path):
        print("creating vocabulary %s from data %s" % (vocabulary_path, data_path))
        vocab={}
        #print('list of directory', os.listdir(data_path))
        for d in os.listdir(data_path):
            if d in ["pos", "neg"]:
                for file in listdir_nohidden(os.path.join(data_path, d)):
                    with gfile.GFile(file, mode= "rb") as f:
                        for line in f:
                            line= tf.compat.as_bytes(line)
                            tokens= basic_tokenizer(line)
                            for t in tokens:
                                if t not in vocab:
                                    vocab[t]=1
                                else:
                                    vocab[t]+=1
                            
                            vocab_list= _START_VOCAB+ sorted(vocab, key= vocab.get, reverse= True)
                            if len(vocab_list)> max_vocabulary_size:
                                vocab_list= vocab_list[:max_vocabulary_size]
                            with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
                                for w in vocab_list:
                                    vocab_file.write(w + b"\n")
            else:
                continue
    print("vocabulary loaded")
                            
def init_vocabulary(vocabulary_path):
    if gfile.Exists(vocabulary_path):
        
        reverse_vocab=[]
        with gfile.GFile(vocabulary_path, mode="rb") as f:
            reverse_vocab.extend(f.readlines())
        reverse_vocab = [tf.compat.as_bytes(line.strip()) for line in reverse_vocab]
        vocab= dict([(y, x) for (x,y) in enumerate(reverse_vocab)])
        return vocab, reverse_vocab
                    
    else:
        raise ValueError("vocabulary file not found in %s directory", vocabulary_path)

def sentence_to_token(sentence, vocabulary):
    words= basic_tokenizer(sentence)
    return [vocabulary.get(w, UNK_ID) for w in words]


def data_to_file(data_path, target_path, vocabulary_path, max_sequence_length):
    #print("fetching data file")
    if not gfile.Exists(target_path):
        print("tokenizing data and storing in %s" % target_path)
        #nb_classes=2
        vocab,_= init_vocabulary(vocabulary_path)
        counter=0
        #Y=[]
        with gfile.GFile(target_path, mode="w") as token_file:
                
                for file in os.listdir(data_path):
                    if file in ["pos", "neg"]:
                        for f in listdir_nohidden(os.path.join(data_path, file)): 
                            with gfile.GFile(f, mode="rb") as data_file:
                                counter+=1
                                if counter % 1000 ==0:
                                    print("reading file %d" % counter)
                                for line in data_file:
                                    token_ids= sentence_to_token(line, vocab)
                                    num_tokens= len(token_ids)
                                    if len(token_ids)<max_sequence_length:
                                        token_ids= token_ids+ [(PAD_ID) for i in range(max_sequence_length- len(token_ids))]
                                        token_ids.append(num_tokens)
                                    else:
                                        token_ids=token_ids[:max_sequence_length]
                                        token_ids.append(max_sequence_length)
                                if "pos" in os.path.join(data_path, file):
                                    token_ids.append(1)
                                else:
                                    token_ids.append(0)
                            
                            token_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")
                    else:
                        continue
                    
                
        
def prepare_data(data_dir, max_vocabulary_size, max_sequence_length):
    vocab_path= os.path.join(data_dir, "vocab%d" % max_vocabulary_size)
    create_vocabulary(vocab_path, data_dir, max_vocabulary_size)
    
    train_ids_path= os.path.join(data_dir, (data_dir + (".ids%d" % max_vocabulary_size)))
    data_to_file(data_dir, train_ids_path, vocab_path, max_sequence_length)
    
    return train_ids_path, vocab_path
    
    
     
        
