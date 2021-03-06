
# coding: utf-8

# In[1]:

import tensorflow as tf
import numpy as np
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import rnn

#Defining some hyper-params
n_hidden = 2       #this is the parameter for input_size in the basic LSTM cell
input_size = 2      #n_hidden and input_size will be the same
embedding_size = 300

batch_size = 50
sentence_length = 55
num_epochs=50


# In[2]:

import data.load as load
from functools import reduce

#train_set, valid_set, test_set, dic = load.atisfold(3)
train_set, test_set, dic = load.atisfull()
idx_pad = max(dic['words2idx'].values()) + 1
dic['words2idx']['<PAD>'] = idx_pad

y_pad = 126

idx2label = dict((v,k) for k,v in dic['labels2idx'].items())
idx2word = dict((v,k) for k,v in dic['words2idx'].items())

train_lex, train_ne, train_y = train_set
#valid_lex, valid_ne, valid_y = valid_set
test_lex,  test_ne,  test_y  = test_set

vocsize = len(set(reduce(lambda x, y: list(x)+list(y),
                         train_lex+test_lex))) + 1 # +1 for padding

nclasses = len(set(reduce(lambda x, y: list(x)+list(y),
                          train_y+test_y)))
nsentences = len(train_lex)


# In[3]:

max_sentence = max([len(s) for s in train_lex])
print(max_sentence)


# In[4]:

print(vocsize)
print(idx_pad)


# In[5]:

len(train_lex) + len(test_lex)


# In[6]:

def padding(sentence, pad=-1, max_length=50):
    length = len(sentence)
    if len(sentence) < max_length:
        sentence = np.append(sentence, [pad] * (max_length - length))
    return sentence


# In[7]:

print(padding(train_lex[0], 46))
print(train_y[0])


# In[8]:

from scipy.sparse import csr_matrix
"""
each batch is a sentence
each batch is a 2D matrix
* height: length of the sentence
* width: the vocaburary size
* x: [n_batch, sentence_length]
* y: [n_batch, sentence_length, nclasses]
"""
def gen_data(source, s_pad, Y, y_pad, max_length, vocsize, nclasses, n_batch=5):
    l = n_batch
    for i in range(len(source)):
        if (i*l+l) >= len(source):
            break
        sentences = source[i*l:i*l+l]
        X = [padding(sentence, s_pad, max_length=max_length) for sentence in sentences]        
        answers = Y[i*l:i*l+l]
        y = []
        for j, answer in enumerate(answers):
            answer = padding(answer, y_pad, max_length=max_length)
            row = np.array([k for k in range(max_length)])
            col = np.array([answer[k] for k in range(max_length)])
            data = np.array([1.0 for _ in range(max_length)])
            m = csr_matrix((data, (row, col)), shape=(max_length, nclasses))
            y.append(m.toarray())
        
        yield (np.array(X), np.array(y))


# In[9]:

def gen_train_data(n_batch):
    g = gen_data(train_lex, idx_pad, train_y, y_pad, max_sentence, vocsize, nclasses, n_batch)
    for i in g:
        yield i
def gen_test_data(n_batch):
    while True:
        g = gen_data(test_lex, idx_pad, test_y, y_pad, max_sentence, vocsize, nclasses, n_batch)
        for i in g:
            yield i


# In[10]:

g = gen_test_data(3)


# In[11]:

X, y = next(g)


# In[12]:

X.shape


# In[13]:

print(y.shape)
print(y[0].shape)


# ### Model Construction

# In[14]:

#create placeholders for X and y

input_x = tf.placeholder(tf.int32, shape=[batch_size, max_sentence])
input_y = tf.placeholder(tf.float32, shape=[batch_size, max_sentence, nclasses])

with tf.device('/cpu:0'), tf.name_scope("embedding"):
    W = tf.Variable(
        tf.random_uniform([vocsize, embedding_size], -1.0, 1.0),
        name="W")
    embedded_chars = tf.nn.embedding_lookup(W, input_x) #


# In[15]:

cell = rnn_cell.BasicLSTMCell(embedding_size)

inputs = tf.split(1, max_sentence, embedded_chars)
inputs = [tf.reshape(i, shape=[batch_size, -1]) for i in inputs]
outputs, states = rnn.rnn(cell, inputs, dtype=tf.float32)

W_o = tf.Variable(tf.random_normal([embedding_size, nclasses], stddev=0.01)) 
b_o = tf.Variable(tf.random_normal([nclasses], stddev=0.01))

outputs3 = [tf.matmul(output, W_o) + b_o for output in outputs] 

y_answers = tf.split(1, max_sentence, input_y)
y_answers2 = [tf.reshape(i, shape=[batch_size, -1]) for i in y_answers]

all_outputs = tf.concat(0, outputs3)
all_answers = tf.concat(0, y_answers2)

losses = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(all_outputs, all_answers))


# In[16]:

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(losses)


# In[17]:

### Generate Validation Data
test_data_gen = gen_test_data(batch_size)


# In[ ]:

X_test, y_test = next(test_data_gen)


# In[ ]:

### Execute
import pdb; 

with tf.Session() as sess:

    tf.initialize_all_variables().run()     #initialize all variables in the model
    
    for k in range(num_epochs):
        
        #Generate Data for each epoch
        #What this does is it creates a list of of elements of length seq_len, each of size [batch_size,input_size]
        #this is required to feed data into rnn.rnn
        train_gen = gen_train_data(50)
        test_gen = gen_test_data(50)
        for tr, te in zip(train_gen, test_gen):
            X = tr[0]
            y = tr[1]
            
            X_test = te[0]
            y_test = te[1]
            # pdb.set_trace()
            #Create the dictionary of inputs to feed into sess.run
            train_dict = {
                input_x: X, # [batch_size, max_sentence]
                input_y: y# [batch_size, max_sentence, nclasses]
            }

            sess.run(optimizer, feed_dict=train_dict)   #perform an update on the parameters

            #create validation dictionary
            test_dict = {
                input_x: X_test, # [batch_size, max_sentence]
                input_y: y_test# [batch_size, max_sentence, nclasses]
            }
            outputs, c_val = sess.run([all_outputs, losses], feed_dict=test_dict)            #compute the cost on the validation set            
            print("Validation cost: {}, on Epoch {}".format(c_val, k))

            if k >= num_epochs-1:
                pred_1st = np.argmax(outputs[::max_sentence], axis=1)
                pred_1st = [idx2label.get(i) for i in pred_1st]

                word_1st = [idx2word.get(i) for i in X_test[0]]
                y_1st = np.argmax(y_test[0], axis=1)
                y_1st = [idx2label.get(i) for i in y_1st]
                # pdb.set_trace()
                wlength = 7
                for w, a, p in zip(word_1st, y_1st, pred_1st):
                    print(w.rjust(wlength), a.rjust(wlength), p.rjust(wlength))
                print('\n'+'**'*30+'\n')


# In[ ]:



