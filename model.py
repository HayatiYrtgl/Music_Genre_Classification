from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import *
from train_test_data import train_dataset, test_dataset
import pandas as pd
import matplotlib.pyplot as plt

def create_model():
    model = Sequential()
    model.add(Input(shape=(28, 128, 1)))
    model.add(Conv2D(filters=128, kernel_size=3, strides=1, padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(11, activation="softmax"))

    model.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999), loss="categorical_crossentropy",
                  metrics=["accuracy"])
    model.summary()
    return model


model = create_model()
history = model.fit(train_dataset, epochs=25, validation_data=test_dataset, batch_size=16)
df = pd.DataFrame(history.history).plot()
plt.show()
model.save("Music_Classifier.h5")

