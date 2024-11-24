# src/trainer.py

import csv
import os
import time

import tensorflow as tf


class Trainer:
    def __init__(self, model, loss='sparse_categorical_crossentropy', optimizer='adam', metrics=None):
        if metrics is None:
            metrics = ['accuracy']
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics

    def compile_model(self):
        self.model.compile(loss=self.loss,
                           optimizer=self.optimizer,
                           metrics=self.metrics)

    def train(self, X_train, y_train, X_val, y_val, run_id, epochs=200, batch_size=20,
              stats_dir='results/per_run_stats'):
        os.makedirs(stats_dir, exist_ok=True)
        run_stats_file = os.path.join(stats_dir, f'run_{run_id}.csv')

        # Создаём файл для записи статистики
        with open(run_stats_file, mode='w', newline='') as csv_file:
            fieldnames = ['epoch', 'loss', 'accuracy', 'val_loss', 'val_accuracy']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            # Callback для записи статистики в CSV
            class EpochLogger(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    writer.writerow({
                        'epoch': epoch + 1,
                        'loss': logs.get('loss'),
                        'accuracy': logs.get('accuracy'),
                        'val_loss': logs.get('val_loss'),
                        'val_accuracy': logs.get('val_accuracy')
                    })
                    csv_file.flush()

            callbacks = [EpochLogger()]

            # Измеряем время обучения
            start_time = time.time()
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                callbacks=callbacks,
                verbose=0  # Логирование эпох выполняется через Callback
            )
            end_time = time.time()
            training_time = end_time - start_time

        print(f"Обучение завершено за {training_time:.2f} секунд. Статистика записана в {run_stats_file}.")
        return history, training_time
