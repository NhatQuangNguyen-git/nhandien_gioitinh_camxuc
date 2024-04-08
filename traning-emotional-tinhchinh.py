from keras.utils import to_categorical
from keras_preprocessing.image import load_img
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from sklearn.preprocessing import LabelEncoder
from tqdm.notebook import tqdm
import os
import pandas as pd
import numpy as np
import optuna
from keras.optimizers import Adam

TRAIN_DIR = '/content/images/train'
TEST_DIR = '/content/images/validation'

def createdataframe(dir):
    image_paths = []
    labels = []
    for label in os.listdir(dir):
        for imagename in os.listdir(os.path.join(dir, label)):
            image_paths.append(os.path.join(dir, label, imagename))
            labels.append(label)
        print(label, "completed")
    return image_paths, labels


train = pd.DataFrame()
train['image'], train['label'] = createdataframe(TRAIN_DIR)

print(train)

test = pd.DataFrame()
test['image'], test['label'] = createdataframe(TEST_DIR)

print(test)
print(test['image'])

from tqdm.notebook import tqdm


def extract_features(images):
    features = []
    for image in tqdm(images):
        img = load_img(image, grayscale=True)
        img = np.array(img)
        features.append(img)
    features = np.array(features)
    features = features.reshape(len(features), 48, 48, 1)
    return features


train_features = extract_features(train['image'])

test_features = extract_features(test['image'])

x_train = train_features / 255.0
x_test = test_features / 255.0

le = LabelEncoder()
le.fit(train['label'])

y_train = le.transform(train['label'])
y_test = le.transform(test['label'])

y_train = to_categorical(y_train, num_classes=7)
y_test = to_categorical(y_test, num_classes=7)


def build_model(trial):
    model = Sequential()
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(7, activation='softmax'))

    lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    opt = Adam(lr=lr)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def objective(trial):
    batch_size = trial.suggest_int("batch_size", 32, 128)

    model = build_model(trial)

    H = model.fit(x_train, y_train, batch_size=batch_size, epochs=100, validation_data=(x_test, y_test), verbose=0)

    _, accuracy = model.evaluate(x_test, y_test)

    return accuracy


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)

best_trial = study.best_trial
print("Best Parameters: ")
print("  lr: ", best_trial.params["lr"])
print("  batch_size: ", best_trial.params["batch_size"])
print("  Accuracy: ", best_trial.value)

# Train the model with the best parameters
best_model = build_model(best_trial)
best_model.fit(x_train, y_train, batch_size=best_trial.params["batch_size"], epochs=100,
               validation_data=(x_test, y_test))