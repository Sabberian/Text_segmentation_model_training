import keras
import segmentation_models as sm
import numpy as np
from sklearn.model_selection import train_test_split

from dataset import Dataset
from cfg import *

ds = Dataset(TRAIN_IMGS_DIR, TRAIN_BINARIES_PATH)

X = []
Y = []
for i in range(len(ds)):
  im, ma = ds[i]
  X.append(im)
  Y.append(ma)
X = np.array(X[:])
Y = np.array(Y[:])
Y = np.expand_dims(Y, axis=3)

x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size = 0.2, random_state=42)

prep = sm.get_preprocessing(NET)
x_train = prep(x_train)
x_val = prep(x_val)

callbacks = [
    keras.callbacks.ModelCheckpoint('./best_model.h5', save_weights_only=True, save_best_only=True, mode='min'),
    keras.callbacks.ReduceLROnPlateau(),
]
model = sm.Unet(NET, classes=1, encoder_weights='imagenet', )
model.compile(optimizer='adam', loss='binary_crossentropy')
history = model.fit(x= x_train,
                    y = y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    callbacks = callbacks,
                    verbose=VERBOSE,
                    validation_data=(x_val, y_val))

