# -*- coding: utf-8 -*-
import random
import numpy as np
from keras import layers
from keras.models import Sequential
from six.moves import range


# 学習データを作成
DIM = 30
INPUT_MAX_LEN = 10
OUTPUT_MAX_LEN = 10
TRAINING_SIZE = 200

x_data = []
y_data = []

for i in range(TRAINING_SIZE + 10):
    # 入力の累積を出力
    s = 0
    xx = []
    yy = []
    for j in range(INPUT_MAX_LEN):
        rnd = random.randint( 0, 3 )
        x = np.zeros( DIM, dtype=np.bool )
        x[rnd] = True   # one-hot voctorにする
        s += rnd
        
        y = np.zeros( DIM, dtype=np.bool )
        y[s] = True   # one-hot voctorにする

        xx.append(x)
        yy.append(y)
        
    x_data.append(xx)
    y_data.append(yy)
        
x_data = np.array(x_data)
y_data = np.array(y_data)


# 学習用と評価用に分割
x_train = x_data[:TRAINING_SIZE]
y_train = y_data[:TRAINING_SIZE]

x_val = x_data[TRAINING_SIZE:]
y_val = y_data[TRAINING_SIZE:]


# 学習データ生成
RNN = layers.LSTM
HIDDEN_SIZE = 256
BATCH_SIZE = 128
LAYERS = 1

# モデル構築
def model():
    m = Sequential()
    from keras.layers.core import Dense, Reshape
    from keras.layers.wrappers import TimeDistributed
    m.add(RNN(HIDDEN_SIZE, input_shape=(INPUT_MAX_LEN, DIM)))
    m.add(Dense(OUTPUT_MAX_LEN * DIM))
    m.add(Reshape((OUTPUT_MAX_LEN, DIM)))
    m.add(TimeDistributed(Dense(DIM, activation='softmax')))
    return m

model = model()
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

# 学習
for iteration in range(1, 200):
    print()
    print('-----------')
    print('Iteration', iteration)
    model.fit(x_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=1,
              validation_data=(x_val, y_val))

# テスト
for i in range(10):
    #ind = np.random.randint(0, len(x_val))
    #rowx, rowy = x_val[np.array([ind])], y_val[np.array([ind])]  # replace by x_val, y_val
    preds = model.predict_classes( np.array([x_val[i]]), verbose=0)
    print( "input: ",  [ np.argmax(x) for x in x_val[i] ] )
    print( "output: ", preds )
    print("--------")
     
