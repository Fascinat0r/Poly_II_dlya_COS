# src/model_builder.py

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential


class ModelBuilder:
    def __init__(self, input_shape=(28, 28, 3), num_classes=9):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build_model(self, num_hidden_layers=1, neurons_per_layer=500):
        model = Sequential()

        # Первый сверточный слой
        model.add(Conv2D(16, (3, 3), input_shape=self.input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Дополнительные сверточные слои (фиксированные)
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())

        # Добавление скрытых полносвязных слоев
        for _ in range(num_hidden_layers):
            model.add(Dense(neurons_per_layer))
            model.add(Activation('relu'))
            model.add(Dropout(0.25))

        # Выходной слой
        model.add(Dense(self.num_classes))
        model.add(Activation('softmax'))

        return model
