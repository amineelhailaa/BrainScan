from cffi import model
from tensorflow.keras import Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Dropout,
    GlobalAveragePooling2D, Dense
)

def create_arch():
    model=  Sequential([
        Input(shape=(224, 224, 3)),

        Conv2D(32, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2)),

        GlobalAveragePooling2D(),
        Dense(128, activation="relu"),
        Dropout(0.5),

        Dense(4, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()
    plot_model(
        model,
        show_shapes=True,
        show_layer_names=True,
        to_file="cnn_architecture.png"
    )
    return model