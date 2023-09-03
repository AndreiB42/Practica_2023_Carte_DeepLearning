import keras
import numpy as np
from keras import layers, models, optimizers, losses, metrics, datasets, utils, losses, regularizers
from keras import Input
from keras.models import compile

callbacks_list = [
keras.callbacks.EarlyStopping(
monitor='accuracy',
patience=1,
),
keras.callbacks.ModelCheckpoint(
filepath='my_model.h5',
monitor='val_loss',
save_best_only=True,
)
]
models.compile(optimizer='rmsprop',
loss='binary_crossentropy',
metrics=['accuracy'])
models.fit(x, y,epochs=10,batch_size=32,callbacks=callbacks_list,validation_data=(x_val, y_val))

callbacks_list = [
keras.callbacks.ReduceLROnPlateau(
monitor='val_loss'
factor=0.1,
patience=10,)]
model.fit(x, y,
epochs=10,
batch_size=32,
callbacks=callbacks_list,
validation_data=(x_val, y_val))


