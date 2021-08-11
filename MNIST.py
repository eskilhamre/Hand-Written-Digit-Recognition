from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
import tensorflowjs as tfjs

# load and reshape data
(X_train, y_train), (X_val, y_val) = mnist.load_data()
X_train = X_train.reshape([-1, 28, 28, 1])
X_val = X_val.reshape([-1, 28, 28, 1])

# normalize pixel vals
X_train = X_train / 255
X_val = X_val / 255

# one-hot encode classes
y_train = keras.utils.to_categorical(y_train)
y_val = keras.utils.to_categorical(y_val)

num_classes = y_train.shape[1]

# initialize model
model = Sequential([
    Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation="relu"),
    MaxPooling2D(),
    Conv2D(15, (3, 3), activation="relu"),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(50, activation="relu"),
    Dense(num_classes, activation="softmax")
])

# train model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=15, batch_size=200, verbose=2)
val_loss, val_acc = model.evaluate(X_val, y_val)
print("Validation loss:", val_loss)
print("Validation accuracy:", val_acc)

# save model as tfjs format
tfjs.converters.save_keras_model(model, 'models')