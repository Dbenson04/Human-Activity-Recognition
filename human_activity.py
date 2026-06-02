# human_activity.py
# Daniel Benson, Joel Peters, Josh Anderson
"""
Python script that leverages deep convolutional neural networks (CNNs), transfer learning, and other modern techniques to classify human activities
from video data that we have collected and preprocessed.
"""

import argparse
import os
import pdb
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import keras
import numpy as np
import cv2
import imageio

ROOT = os.path.dirname(os.path.abspath(__file__))
TRAINDIR = os.path.join(ROOT, "data", "samples", "Ref_samples", "Train")
TESTDIR = os.path.join(ROOT, "data", "samples", "Ref_samples", "Test")

CLASSES = ["Ball", "Eject", "Safe", "Time"]
NUM_FRAMES = 40
IMG_SIZE = 128
EPOCHS = 30
BATCH_SIZE = 8

# not actually used anywhere anymore, was being used to load but was giving problems so now its just a link to the model incase we need it
RESNET_URL = "https://www.kaggle.com/models/google/resnet-v2/TensorFlow2/101-feature-vector/2"

parser = argparse.ArgumentParser(description="Transfer learning with ResNet-V2 to classify referee signs.")
parser.add_argument("mode", choices=["train", "test"], help="Run mode")
parser.add_argument("--debug", action="store_true", help="Drop into pdb at end of main")
parser.add_argument('--file', type=str, default=None, help='Path to the video you want to test (e.g., eicholtz_1.gif)')


def build_model():
    # transfer learning with resnet
    base = keras.applications.ResNet50V2(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        pooling="avg",
    )
    base.trainable = False

    inputs = keras.Input(shape=(NUM_FRAMES, IMG_SIZE, IMG_SIZE, 3))
    x = keras.layers.TimeDistributed(base)(inputs) # this thing allows us to use resnet to each of our frames from each sample which is cool
    x = keras.layers.GlobalAveragePooling1D()(x) 
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.4)(x)
    outputs = keras.layers.Dense(len(CLASSES), activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def process_gif(path, max_frames=40, size=(128, 128)):
    # uses imageio to read the gif first because cv2 was having some problems
    gif = imageio.mimread(path)  # reads all frames
    frames = []

    for frame in gif:
        # Resize
        frame = cv2.resize(frame, size)

        # Convert to grayscale
        # resnet prefers color images so not using this here
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Normalize
        frame = frame / 255.0

        frames.append(frame)

    # Handle empty case safely
    if len(frames) == 0:
        raise ValueError(f"No frames found in GIF: {path}")

    # Trim or pad
    # makes sure that all gifs end up as the same length because the neural network
    # requires all of the inputs to have the same size
    frames = frames[:max_frames]
    while len(frames) < max_frames:
        frames.append(frames[-1])

    frames = np.array(frames)

    return frames

def data_loader(dir):
    x = [] # array of frames extracted from gifs
    y = [] # array of labels for each of the frames
    
    for label, activity in enumerate(CLASSES): # loop through every class in inputted directory (train, test)
        current_activity =  os.path.join(dir, activity)
        
        for file in os.listdir(current_activity): # for each class, go through each file (gif) and grab it's path - then call process_gif(path) to extract frames
            file_path = os.path.join(current_activity, file)
            frames = process_gif(file_path)
            
            x.append(frames)
            y.append(label)
    
    return np.array(x), np.array(y)

def main():
    args = parser.parse_args()

    if args.mode == "train":
        X, Y = data_loader(TRAINDIR)
        x_test, y_test = data_loader(TESTDIR)
        x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=1/3, random_state=67)

        # Expected: (num_samples, frames, H, W, 1)
        # Actual: (12, 40, 128, 128, 3)
        print(x_train.shape)
        print(x_test.shape)

        if args.debug:
            pdb.set_trace()

        model = build_model()
        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=12,            # start small
            batch_size=10         # small batch size for videos
        )
        
        # these save the whole model to file but the files are ~90mb
        # so its better to not save these to the repo when possible
        # model.save("model_1.keras")
        # model.save_weights("model_weights_1.weights.h5")
        

        model.summary()

        # just a test to make sure its alive
        # sample = np.expand_dims(ar[1], axis=0)

        # print(x_test)
        print(y_test)
        prediction = model.predict(x_test)
        y_pred = np.argmax(prediction, axis=1)
        cm = confusion_matrix(y_test, y_pred)
        print(prediction)
        print(cm)
    
    
    # lets you pass in an individual file to run through the model and get its prediction
    # i think this still takes a while each time because of the Resnet step
    if args.mode == "test":
        model = keras.models.load_model("model_2.keras")
        
        if not os.path.exists(args.file):
            print("Error: File not found")
            return
        
        data = process_gif(args.file)
        data = np.expand_dims(data, axis=0)
        prediction = model.predict(data)
        print(prediction)

    if args.debug:
        pdb.set_trace()


if __name__ == "__main__":
    main()