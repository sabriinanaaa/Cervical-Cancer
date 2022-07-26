from imutils import paths
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from Model.Colponet import ColpoNet
import os
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)

args = {"train": "train",
        "type1_ad": "additional_Type_1_v2",
        "type2_ad": "additional_Type_2_v2",
        "type3_ad": "additional_Type_3_v2",
        "epochs": 300,
        "plots": "plots"}
print("[INFO] loading images...")

imagePaths = list(paths.list_images(args["train"]))
print("total train set:", len(imagePaths))

type1_ad = list(paths.list_images(args["type1_ad"]))
type2_ad = list(paths.list_images(args["type2_ad"]))
type3_ad = list(paths.list_images(args["type3_ad"]))
print("additional set: type 1 = ", len(type1_ad),
      ", type 2 = ", len(type2_ad), ", type 3 = ", len(type3_ad))

data = []
labels = []

for imagePath in imagePaths:

    label = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    print(f"image={imagePath.split(os.path.sep)[-1]}")
    image = cv2.resize(image, (64, 64))

    data.append(image)
    labels.append(label)

# trainX(list): training set, trainY(list): training set label
# testX(list): validation set, testY(list): validation set label
(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=0.30,
                                                  stratify=labels,
                                                  random_state=42)
print('Train:', len(trainX), '\nVal:', len(testY))

for additionalPath in type1_ad:

    label1 = additionalPath.split(os.path.sep)[-2]
    trainY.append(label1)

    image1 = cv2.imread(additionalPath)
    print(f"image={additionalPath.split(os.path.sep)[-1]}")
    image1 = cv2.resize(image1, (64, 64))
    trainX.append(image1)

for additionalPath2 in type2_ad:

    label2 = additionalPath2.split(os.path.sep)[-2]
    trainY.append(label2)

    image2 = cv2.imread(additionalPath2)
    print(f"image={additionalPath2.split(os.path.sep)[-1]}")
    image2 = cv2.resize(image2, (64, 64))
    trainX.append(image2)

for additionalPath in type3_ad:

    label3 = additionalPath.split(os.path.sep)[-2]
    trainY.append(label3)

    image3 = cv2.imread(additionalPath)
    print(f"image={additionalPath.split(os.path.sep)[-1]}")
    image3 = cv2.resize(image3, (64, 64))
    trainX.append(image3)
    
print(f'Train={len(trainX)}\nVal={len(testY)}')

# type list -> type numpy.ndarray
trainX = np.array(trainX, dtype="float") / 255.0
testX = np.array(testX, dtype="float") / 255.0

le = LabelEncoder()
trainY = le.fit_transform(trainY)
trainY = to_categorical(trainY, 3)
testY = le.fit_transform(testY)
testY = to_categorical(testY, 3)
print(f'Train={len(trainX)},{type(trainX)}\nVal={len(testY)},{type(testY)}')

# random crop
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.5,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.5,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest")

# build the model - ColpoNet
model = ColpoNet.build(width=64, height=64, depth=3,
                        filters=32, classes=3, reg=0.0002)

opt = Adam(lr=1e-4, decay=1e-4 / args["epochs"])
# opt = SGD(lr=1e-5, momentum=0.9, decay=1e-5 / args["epochs"])
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

print("[INFO] training network for {} epochs...".format(args["epochs"]))
startTime = time.time()
BS = 32
EPOCHS = args["epochs"]
H = model.fit(
    x=aug.flow(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // BS,
    epochs=args["epochs"])
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))



# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1), target_names=le.classes_))


# plot the training loss and accuracy
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["acc"], label="train_acc")
plt.plot(N, H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plots"])