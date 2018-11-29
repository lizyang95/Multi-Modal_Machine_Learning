import numpy as np
import tensorflow as tf
import chakin

import json
import os
from collections import defaultdict

from model.ass_fun import *

chakin.search(lang='English')

CHAKIN_INDEX = 14
NUMBER_OF_DIMENSIONS = 300
SUBFOLDER_NAME = "glove.6B.300d"

gpu = cfg.GPU
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# download Twitter.25d embeddings from Standford
DATA_FOLDER = "embeddings"
ZIP_FILE = os.path.join(DATA_FOLDER, "{}.zip".format(SUBFOLDER_NAME))
ZIP_FILE_ALT = "glove" + ZIP_FILE[5:]  # sometimes it's lowercase only...
UNZIP_FOLDER = os.path.join(DATA_FOLDER, SUBFOLDER_NAME)
if SUBFOLDER_NAME[-1] == "d":
    GLOVE_FILENAME = os.path.join(UNZIP_FOLDER, "{}.txt".format(SUBFOLDER_NAME))
else:
    GLOVE_FILENAME = os.path.join(UNZIP_FOLDER, "{}.{}d.txt".format(SUBFOLDER_NAME, NUMBER_OF_DIMENSIONS))


if not os.path.exists(ZIP_FILE) and not os.path.exists(UNZIP_FOLDER):
    # GloVe by Stanford is licensed Apache 2.0:
    #     https://github.com/stanfordnlp/GloVe/blob/master/LICENSE
    #     http://nlp.stanford.edu/data/glove.twitter.27B.zip
    #     Copyright 2014 The Board of Trustees of The Leland Stanford Junior University
    print("Downloading embeddings to '{}'".format(ZIP_FILE))
    chakin.download(number=CHAKIN_INDEX, save_dir='./{}'.format(DATA_FOLDER))
else:
    print("Embeddings already downloaded.")

if not os.path.exists(UNZIP_FOLDER):
    import zipfile
    if not os.path.exists(ZIP_FILE) and os.path.exists(ZIP_FILE_ALT):
        ZIP_FILE = ZIP_FILE_ALT
    with zipfile.ZipFile(ZIP_FILE,"r") as zip_ref:
        print("Extracting embeddings to '{}'".format(UNZIP_FOLDER))
        zip_ref.extractall(UNZIP_FOLDER)
else:
    print("Embeddings already extracted.")


def load_embedding_from_disks(glove_filename, with_indexes=True):
    """
    Read a GloVe txt file. If `with_indexes=True`, we return a tuple of two dictionnaries
    `(word_to_index_dict, index_to_embedding_array)`, otherwise we return only a direct
    `word_to_embedding_dict` dictionnary mapping from a string to a numpy array.
    """
    if with_indexes:
        word_to_index_dict = dict()
        index_to_embedding_array = []
    else:
        word_to_embedding_dict = dict()

    print("finish index\n")
    with open(glove_filename, 'r') as glove_file:
        print("file opened\n")
        for (i, line) in enumerate(glove_file):

            split = line.split(' ')

            word = split[0]

            representation = split[1:]
            representation = np.array(
                [float(val) for val in representation]
            )

            if with_indexes:
                word_to_index_dict[word] = i
                index_to_embedding_array.append(representation)
            else:
                word_to_embedding_dict[word] = representation

    print("find word\ns")
    _WORD_NOT_FOUND = [0.0]* len(representation)  # Empty representation for unknown words.
    if with_indexes:
        _LAST_INDEX = i + 1
        word_to_index_dict = defaultdict(lambda: _LAST_INDEX, word_to_index_dict)
        index_to_embedding_array = np.array(index_to_embedding_array + [_WORD_NOT_FOUND])
        return word_to_index_dict, index_to_embedding_array
    else:
        word_to_embedding_dict = defaultdict(lambda: _WORD_NOT_FOUND)
        return word_to_embedding_dict

# test loading embedding
print("Loading embedding from disks...")
word_to_index, index_to_embedding = load_embedding_from_disks(GLOVE_FILENAME, with_indexes=True)
print("Embedding loaded from disks.")

vocab_size, embedding_dim = index_to_embedding.shape
print("Embedding is of shape: {}".format(index_to_embedding.shape))
print("This means (number of words, number of dimensions per word)\n")
print("The first words are words that tend occur more often.")

print("Note: for unknown words, the representation is an empty vector,\n"
      "and the index is the last one. The dictionnary has a limit:")
print("    {} --> {} --> {}".format("A word", "Index in embedding", "Representation"))
word = "worsdfkljsdf"
idx = word_to_index[word]
embd = list(np.array(index_to_embedding[idx], dtype=int))  # "int" for compact print only.
print("    {} --> {} --> {}".format(word, idx, embd))
word = "the"
idx = word_to_index[word]
embd = list(index_to_embedding[idx])  # "int" for compact print only.
print("    {} --> {} --> {}".format(word, idx, embd))


# load the embedding in tensorflow

batch_size = None  # Any size is accepted

tf.reset_default_graph()
sess = tf.InteractiveSession()  # sess = tf.Session()

# Define the variable that will hold the embedding:
tf_embedding = tf.Variable(
    tf.constant(0.0, shape=index_to_embedding.shape),
    trainable=False,
    name="Embedding"
)

tf_word_ids = tf.placeholder(tf.int32, shape=[batch_size])

tf_word_representation_layer = tf.nn.embedding_lookup(
    params=tf_embedding,
    ids=tf_word_ids
)
tf_embedding_placeholder = tf.placeholder(tf.float32, shape=index_to_embedding.shape)
tf_embedding_init = tf_embedding.assign(tf_embedding_placeholder)
_ = sess.run(
    tf_embedding_init,
    feed_dict={
        tf_embedding_placeholder: index_to_embedding
    }
)

print("Embedding now stored in TensorFlow. Can delete numpy array to clear some CPU RAM.")
del index_to_embedding


# test on a batch of word
dic = load_dic()
batch_of_words = list(dic.values())
batch_indexes = [word_to_index[w.lower()] for w in batch_of_words]

embedding_from_batch_lookup = sess.run(
    tf_word_representation_layer,
    feed_dict={
        tf_word_ids: batch_indexes
    }
)
print("Representations for {}:".format(batch_of_words))
print(embedding_from_batch_lookup)

# save word_to_index
prefix = SUBFOLDER_NAME + "." + str(NUMBER_OF_DIMENSIONS) + "d"
TF_EMBEDDINGS_FILE_NAME = os.path.join(DATA_FOLDER, prefix + ".ckpt")
DICT_WORD_TO_INDEX_FILE_NAME = os.path.join(DATA_FOLDER, prefix + ".json")

variables_to_save = [tf_embedding]
embedding_saver = tf.train.Saver(variables_to_save)
embedding_saver.save(sess, save_path=TF_EMBEDDINGS_FILE_NAME)
print("TF embeddings saved to '{}'.".format(TF_EMBEDDINGS_FILE_NAME))
sess.close()

# with open(DICT_WORD_TO_INDEX_FILE_NAME, 'w') as f:
#     json.dump(word_to_index, f)
# print("word_to_index dict saved to '{}'.".format(DICT_WORD_TO_INDEX_FILE_NAME))
