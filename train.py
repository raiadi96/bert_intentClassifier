import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
from output_utils import plot_train_history


def train_model(model, data):
    save_callback = create_save_checkpoint()
    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")]
    )

    history = model.fit(
        x=data.train_x,
        y=data.train_y,
        validation_split=0.1,
        batch_size=16,
        shuffle=True,
        epochs=1,
        callbacks=[save_callback]
    )

    plot_train_history(history)

def create_save_checkpoint():
    path_dir = 'checkpoint'
    file_name = 'intent_bert.ckpt'

    return tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(path_dir, file_name),\
                                                     save_weights_only=True,\
                                                     verbose=1)