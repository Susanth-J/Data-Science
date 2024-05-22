import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)


zs = tf.keras.models.Sequential()
zs.add(tf.keras.layers.Flatten(input_shape=(28,28)))
zs.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
zs.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
zs.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
zs.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

zs.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

zs.fit(x_train, y_train, epochs=3)

accuracy, loss =zs.evaluate(x_test, y_test)

print(accuracy)
print(loss)

zs.save("digits.zs")

for x in range(1,2):
    image_path = os.path.join("/Users/susanthjagarlamudi/Documents/code/VS/Python-Prac/Projects/Number Recognisiton/1.png", "1.png")
    img = cv.imread(image_path)[:,:,0]
    img = np.invert(np.array([img]))
    prediction =zs.pridict(img)
    print(f"the probability is : {np.argmax(prediction)}")
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()