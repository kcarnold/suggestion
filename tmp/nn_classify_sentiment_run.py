# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 15:26:16 2017

@author: kcarnold
"""

import numpy as np
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
import joblib

#%%
preproc = joblib.load('keras_training_data.pkl')
x_train = preproc['x_train']
y_train = preproc['y_train']
x_val = preproc['x_test']
y_val = preproc['y_test']
id2str = preproc['id2str']
embedding_mat = preproc['embedding_mat']
num_words, EMBEDDING_DIM = embedding_mat.shape
#%%
num_labels = np.max(y_train) + 1

#%%
MAX_SEQ_LEN = x_train.shape[1]
#%%
embedding_mat[0] = 0
#%%
embedding_layer = Embedding(num_words, EMBEDDING_DIM,
                            # weights=[embedding_mat],
                            input_length=MAX_SEQ_LEN,
                            trainable=True)

y_train_categorical = np.eye(num_labels)[y_train-1]
y_val_categorical = np.eye(num_labels)[y_val-1]


# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQ_LEN,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(256, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(2)(x)
#x = Conv1D(128, 2, activation='relu')(x)
#x = MaxPooling1D(2)(x)
#x = Conv1D(128, 5, activation='relu')(x)
#x = MaxPooling1D(35)(x)
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
preds = Dense(num_labels, activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'],
)

model.fit(x_train, y_train_categorical,
          batch_size=128,
          epochs=10,
          validation_data=(x_val, y_val_categorical),
          callbacks=[
                  ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=True),
                  EarlyStopping('val_loss', patience=0, verbose=1),
                  # TensorBoard(log_dir='./tensorboard-logs', histogram_freq=1, write_grads=True, embeddings_freq=1)
                  ])
