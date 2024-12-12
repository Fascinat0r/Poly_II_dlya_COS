# src/experiment_runner.py

import csv
import os
from datetime import datetime

from lab11.src.data_generator import DataGenerator
from lab11.src.evaluator import Evaluator
from lab11.src.haralick_data_loader import HaralickDataLoader
from lab11.src.model_builder import ModelBuilder
from lab11.src.trainer import Trainer


class ExperimentRunner:
    def __init__(self, hyperparameters, results_file='results/experiment_results.csv',
                 haralick_params_file='data/haralick_parameters.csv', stats_dir='results/per_run_stats'):
        self.hyperparameters = hyperparameters
        self.results_file = results_file
        self.haralick_params_file = haralick_params_file
        self.stats_dir = stats_dir
        self.epochs = 100
        self.batch_size = 20

        os.makedirs(os.path.dirname(self.results_file), exist_ok=True)
        os.makedirs(self.stats_dir, exist_ok=True)

    def load_existing_runs(self):
        """
        Загружает список уже выполненных прогонов из общего CSV файла.
        """
        existing_runs = set()
        if os.path.exists(self.results_file):
            with open(self.results_file, 'r', newline='', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    run_key = (
                        row['model_type'],
                        row['colors_used'],
                        int(row['num_train_samples'])
                    )
                    existing_runs.add(run_key)
        return existing_runs

    def run_experiments(self):
        # Загрузка параметров Харалика
        haralick_params = HaralickDataLoader.get_haralick_params(self.haralick_params_file)
        # color_to_params = HaralickDataLoader.group_params_by_color(haralick_params)

        # Загрузка существующих прогонов
        existing_runs = self.load_existing_runs()

        # Столбцы для результатов
        fieldnames = ['model_type', 'colors_used', 'num_train_samples', 'epochs', 'batch_size',
                      'train_time_sec', 'test_loss', 'test_accuracy']
        with open(self.results_file, 'a', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if not existing_runs:  # Если файл пустой, записываем заголовок
                writer.writeheader()

            for model_type in self.hyperparameters['model_types']:
                for color_combination in self.hyperparameters['color_combinations']:
                    colors_used_str = ','.join(color_combination)
                    for num_train_samples in self.hyperparameters['num_samples_list']:
                        run_key = (model_type, colors_used_str, num_train_samples)
                        if run_key in existing_runs:
                            print(f"Пропуск существующего прогона: {run_key}")
                            continue

                        print(
                            f"Запуск эксперимента: {model_type}, цветов: {color_combination}, {num_train_samples} образцов.")

                        # Формирование уникального run_id
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        run_id = f"{model_type}_{'_'.join(color_combination)}_{num_train_samples}_{timestamp}"

                        # Инициализация DataGenerator с фиксированной формой данных
                        data_gen = DataGenerator(
                            num_samples_per_disease=num_train_samples // 8,
                            haralick_params=haralick_params,
                            color_combination=color_combination,
                            model_type=model_type
                        )

                        (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_gen.generate_all_datasets(
                            num_train_samples)

                        # Получение формы данных
                        input_shape = X_train.shape[1:]  # (24,) для FFNN или (4,6,3) для CNN
                        print(f"Используемая форма входных данных для модели {run_id}: {input_shape}")

                        # Построение модели
                        model_builder = ModelBuilder(input_shape=input_shape, model_type=model_type)
                        model = model_builder.build_model()

                        # Обучение модели
                        trainer = Trainer(model)
                        trainer.compile_model()
                        history, train_time = trainer.train(
                            X_train, y_train, X_val, y_val, run_id,
                            epochs=self.epochs,
                            batch_size=self.batch_size,
                            stats_dir=self.stats_dir
                        )

                        # Оценка модели
                        evaluator = Evaluator(model)
                        test_loss, test_accuracy = evaluator.evaluate(X_test, y_test)

                        # Сохранение результатов
                        writer.writerow({
                            'model_type': model_type,
                            'colors_used': colors_used_str,
                            'num_train_samples': num_train_samples,
                            'epochs': self.epochs,
                            'batch_size': self.batch_size,
                            'train_time_sec': train_time,
                            'test_loss': test_loss,
                            'test_accuracy': test_accuracy
                        })
                        csv_file.flush()
                        print(
                            f"Эксперимент завершён: {model_type}, цветов: {color_combination}, {num_train_samples} образцов. "
                            f"run_id: {run_id}")
