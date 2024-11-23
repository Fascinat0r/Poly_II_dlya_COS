# src/hyperparameter_tuner.py

import csv
import itertools
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import psutil

from src.evaluator import Evaluator
from src.trainer import Trainer


class HyperparameterTuner:
    def __init__(self, model_builder, data_loader, trainer, evaluator,
                 results_file='results/hyperparameter_results.csv', stats_dir='results/per_run_stats'):
        self.model_builder = model_builder
        self.data_loader = data_loader
        self.trainer = trainer
        self.evaluator = evaluator
        self.results_file = results_file
        self.stats_dir = stats_dir
        os.makedirs(self.stats_dir, exist_ok=True)

    def single_run(self, hp, run_id):
        num_hidden_layers, neurons_per_layer, epochs, batch_size = hp
        print(
            f"[Run {run_id}] Тренировка с параметрами: Скрытые слои={num_hidden_layers}, Нейроны={neurons_per_layer}, Эпохи={epochs}, Размер батча={batch_size}")

        # Обновление размера батча в загрузчике данных
        self.data_loader.batch_size = batch_size

        # Получение генераторов данных
        train_gen = self.data_loader.get_train_generator()
        val_gen = self.data_loader.get_val_generator()
        test_gen = self.data_loader.get_test_generator()

        # Построение и компиляция модели
        model = self.model_builder.build_model(num_hidden_layers=num_hidden_layers,
                                               neurons_per_layer=neurons_per_layer)
        local_trainer = Trainer(model=model,
                                loss=self.trainer.loss,
                                optimizer=self.trainer.optimizer,
                                metrics=self.trainer.metrics)
        local_trainer.compile_model()

        # Расчет количества шагов
        steps_per_epoch = train_gen.samples // batch_size
        validation_steps = val_gen.samples // batch_size

        # Обучение модели
        history, training_time = local_trainer.train(train_gen, val_gen, epochs, steps_per_epoch, validation_steps,
                                                     run_id, stats_dir=self.stats_dir)

        # Оценка на тестовых данных
        local_evaluator = Evaluator(model=model)
        scores = local_evaluator.evaluate(test_gen, steps=test_gen.samples // batch_size)

        # Подготовка результатов
        result = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'num_hidden_layers': num_hidden_layers,
            'neurons_per_layer': neurons_per_layer,
            'epochs': epochs,
            'batch_size': batch_size,
            'train_loss': history.history['loss'][-1],
            'train_accuracy': history.history['accuracy'][-1],
            'val_loss': history.history['val_loss'][-1],
            'val_accuracy': history.history['val_accuracy'][-1],
            'test_loss': scores[0],
            'test_accuracy': scores[1],
            'training_time_sec': training_time
        }

        print(f"[Run {run_id}] Завершено: {result}")
        return result

    @staticmethod
    def calculate_max_workers(memory_per_process_gb):
        """Определяет максимальное количество процессов, основываясь на доступной памяти."""
        available_memory_gb = psutil.virtual_memory().available / (1024 ** 3)  # в ГБ
        max_workers_by_memory = int(available_memory_gb // memory_per_process_gb)
        max_workers_by_cpu = os.cpu_count() or 1
        return max(1, min(max_workers_by_memory, max_workers_by_cpu))  # Минимум 1 процесс

    def tune(self, hyperparameters, memory_per_process_gb=1.0):
        """
        Запускает подбор гиперпараметров с учётом доступной памяти и процессоров.
        :param hyperparameters: Словарь с параметрами для подбора
        :param memory_per_process_gb: Среднее количество памяти на один процесс (в ГБ)
        """
        # Создание директории для результатов, если она не существует
        os.makedirs(os.path.dirname(self.results_file), exist_ok=True)

        # Определение общего количества комбинаций
        combinations = list(itertools.product(
            hyperparameters['num_hidden_layers'],
            hyperparameters['neurons_per_layer'],
            hyperparameters['epochs'],
            hyperparameters['batch_size']
        ))
        total_runs = len(combinations)
        print(f"Общее количество прогонов: {total_runs}")

        # Открытие CSV файла и запись заголовка
        with open(self.results_file, mode='w', newline='') as csv_file:
            fieldnames = ['timestamp', 'num_hidden_layers', 'neurons_per_layer', 'epochs', 'batch_size',
                          'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy',
                          'test_loss', 'test_accuracy', 'training_time_sec']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            # Рассчитываем количество доступных процессов
            max_workers = self.calculate_max_workers(memory_per_process_gb)
            print(f"Используется {max_workers} рабочих процессов (учёт CPU и RAM).")

            # Использование ProcessPoolExecutor для параллельного выполнения прогонов
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for run_id, hp in enumerate(combinations, start=1):
                    futures.append(executor.submit(self.single_run, hp, run_id))

                for future in as_completed(futures):
                    result = future.result()
                    writer.writerow(result)
                    csv_file.flush()
