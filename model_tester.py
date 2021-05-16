from keras.datasets import mnist
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from PIL import Image

model = load_model('saved_model/my_model')
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
image_index = 8888
image =Image.open('conv_img.bmp')
rawData = image.load()
#print(rawData[0,0])
data = []
for y in range(28):
    row = []
    for x in range(28):
        row.append(255-rawData[x,y])
    data.append(row)

for x in range(len(data)):
    data[0][x] = 0
    data[x][0] = 0
    data[x][1] = 0
    if data[x][1] > 0 and data[x][1] <= 255:
        print("what the hell")
        data[x][1] == 0
data = np.array(data)
plt.imshow(data.reshape(28,28),cmap='Greys')
#plt.imshow(x_test[image_index].reshape(28,28),cmap='Greys')
plt.show()
#print(data)
pred = model.predict(data.reshape(1, 28, 28, 1))
#pred2 = model.predict(x_test[image_index].reshape(1,28,28,1))
print(pred.argmax())
#print(pred2.argmax())