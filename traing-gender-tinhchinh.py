from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
import glob
import optuna

# initial parameters
epochs = 100
img_dims = (96, 96, 3)

data = []
labels = []

# load image files from the dataset
image_files = [f for f in glob.glob(r'/content/Gender-Detection/gender_dataset_face' + "/**/*", recursive=True) if not os.path.isdir(f)]
random.shuffle(image_files)

# converting images to arrays and labelling the categories
for img in image_files:
    image = cv2.imread(img)
    image = cv2.resize(image, (img_dims[0], img_dims[1]))
    image = img_to_array(image)
    data.append(image)

    label = img.split(os.path.sep)[-2]
    if label == "woman":
        label = 1
    else:
        label = 0

    labels.append(label)

# pre-processing
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# split dataset for training and validation
trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.2, random_state=42)
trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)

# augmenting dataset
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")

# define model
def build_model(trial):
    model = Sequential()
    inputShape = (img_dims[0], img_dims[1], img_dims[2])
    chanDim = -1

    if K.image_data_format() == "channels_first":
        inputShape = (img_dims[2], img_dims[0], img_dims[1])
        chanDim = 1

    model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(2))
    model.add(Activation("sigmoid"))

    return model

# define the objective function for Optuna
def objective(trial):
    # generate the hyperparameters to search
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    batch_size = trial.suggest_int("batch_size", 32, 128)

    # build the model
    model = build_model(trial)

    # compile the model
    opt = Adam(lr=lr)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    # train the model
    H = model.fit_generator(
        aug.flow(trainX, trainY, batch_size=batch_size),
        validation_data=(testX, testY),
        steps_per_epoch=len(trainX) // batch_size,
        epochs=epochs,
        verbose=0
    )

    # evaluate the model
    _, accuracy = model.evaluate(testX, testY)

    return accuracy

# create a study object and optimize the objective function
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

# get the best trial and print the best parameters
best_trial = study.best_trial
print("Best Parameters: ")
print("  lr: ", best_trial.params["lr"])
print("  batch_size: ", best_trial.params["batch_size"])
print("  Accuracy: ", best_trial.value)

# build and compile the model with the best parameters
best_model = build_model(best_trial)
best_opt = Adam(lr=best_trial.params["lr"])
best_model.compile(loss="binary_crossentropy", optimizer=best_opt, metrics=["accuracy"])

# train the model with the best parameters
best_H = best_model.fit_generator(
    aug.flow(trainX, trainY, batch_size=best_trial.params["batch_size"]),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // best_trial.params["batch_size"],
    epochs=epochs,
    verbose=1
)

# save the model to disk
best_model.save('gender_detection.model')

# plot training/validation loss/accuracy
plt.style.use("ggplot")
plt.figure()
N = epochs
plt.plot(np.arange(0, N), best_H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), best_H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), best_H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), best_H.history["val_accuracy"], label="val_acc")

plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")

# save plot to disk
plt.savefig('plot.png')