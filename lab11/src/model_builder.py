# model_builder.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout

class ModelBuilder:
    def __init__(self, input_shape=(4, 6, 3), num_classes=8):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build_model(self):
        model = Sequential()
        model.add(Flatten(input_shape=self.input_shape))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(self.num_classes, activation='softmax'))
        return model
