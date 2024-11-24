# src/trainer.py

import csv
import os
import time

import tensorflow as tf


class Trainer:
    def __init__(self, model, loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy']):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics

    def compile_model(self):
        self.model.compile(loss=self.loss,
                           optimizer=self.optimizer,
                           metrics=self.metrics)

    def train(self, train_generator, val_generator, epochs, steps_per_epoch, validation_steps, run_id,
              stats_dir='results/per_run_stats'):
        os.makedirs(stats_dir, exist_ok=True)
        run_stats_file = os.path.join(stats_dir, f'run_{run_id}.csv')

        # Открытие CSV файла для записи статистики по эпохам
        with open(run_stats_file, mode='w', newline='') as csv_file:
            fieldnames = ['epoch', 'loss', 'accuracy', 'val_loss', 'val_accuracy']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            # Функция обратного вызова для записи статистики
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

            # Измерение времени обучения
            start_time = time.time()
            history = self.model.fit(
                train_generator,
                steps_per_epoch=steps_per_epoch,
                validation_data=val_generator,
                validation_steps=validation_steps,
                epochs=epochs,
                shuffle=True,
                callbacks=callbacks,
                verbose=0
            )
            end_time = time.time()
            training_time = end_time - start_time

        return history, training_time
