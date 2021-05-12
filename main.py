from keras.datasets import mnist
from matplotlib import pyplot
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
# load dataset
(trainX, trainy), (testX, testy) = mnist.load_data()


#reshaping the array to 4 dimenstional numpy array, used for Keras API
trainX = trainX.reshape(trainX.shape[0],28,28,1)
testX = testX.reshape(testX.shape[0],28,28,1)
inp_shape = (28,28,1)
trainX = trainX.astype('float32')
testX = testX.astype('float32')
#normalize RGB values
trainX /= 255
testX /= 255
print("trainX shape: ", trainX.shape)
#setting up the model
model = Sequential()
model.add(Conv2D(28,kernel_size=(3,3),input_shape=inp_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128,activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
model.fit(x=trainX,y=trainy, epochs=10)

model.evaluate(trainX, testy)