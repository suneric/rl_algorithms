import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

(x_train,y_train),(x_test,y_test) = keras.datasets.mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

print(x_train.shape)

model = Sequential()
model.add(LSTM(64,input_shape=(x_train.shape[1:]),return_sequences=True,activation='relu'))
model.add(Dropout(0.2))
model.add(LSTM(64,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10,activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(lr=1e-3),
    metrics=['accuracy'])
model.summary()
# model.fit(x_train,y_train,epochs=3,validation_data=(x_test,y_test))
