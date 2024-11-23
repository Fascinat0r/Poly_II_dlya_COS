# trainer.py

import time

class Trainer:
    def __init__(self, model, loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy']):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics

    def compile_model(self):
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

    def train(self, X_train, y_train, X_val, y_val, epochs=200, batch_size=20):
        start_time = time.time()
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            verbose=0
        )
        training_time = time.time() - start_time
        return history, training_time
