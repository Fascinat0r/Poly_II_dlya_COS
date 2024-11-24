# src/model_builder.py

import math

from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential


class ModelBuilder:
    def __init__(self, input_shape, num_classes=8, model_type='FFNN'):
        """
        :param input_shape: Shape of the input data.
        :param num_classes: Number of output classes.
        :param model_type: Type of model to build ('FFNN' or 'CNN').
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_type = model_type

    def build_model(self):
        if self.model_type == 'FFNN':
            return self.build_ffnn()
        elif self.model_type == 'CNN':
            if len(self.input_shape) != 3:
                raise ValueError(f"Для CNN требуется 3D входные данные, получено: {self.input_shape}")
            if self.input_shape[0] < 3 or self.input_shape[1] < 3:
                raise ValueError(f"Размер входных данных слишком мал для CNN: {self.input_shape}")
            return self.build_cnn()
        else:
            raise ValueError(f"Неизвестный тип модели: {self.model_type}")

    def build_ffnn(self):
        model = Sequential()
        model.add(Flatten(input_shape=self.input_shape))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(self.num_classes, activation='softmax'))
        return model

    def build_cnn(self):
        """
        Построение CNN модели с учётом небольшого размера входных данных.
        """
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=self.input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), padding='same'))  # Поддерживаем минимальные размеры
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), padding='same'))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation='softmax'))
        return model

    @staticmethod
    def _calculate_output_size(input_size, kernel_size=3, pool_size=2):
        """
        Рассчитывает размеры данных после свёртки и пуллинга.

        :param input_size: Tuple (height, width) входных данных.
        :param kernel_size: Размер ядра свёртки.
        :param pool_size: Размер ядра пуллинга.
        :return: Tuple (height, width) выходных данных.
        """
        height, width = input_size
        height = math.floor((height - (kernel_size - 1)) / pool_size)
        width = math.floor((width - (kernel_size - 1)) / pool_size)
        return height, width
