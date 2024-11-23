# experiment_runner.py

import csv
import os

from lab11.src.data_generator import DataGenerator
from lab11.src.evaluator import Evaluator
from lab11.src.model_builder import ModelBuilder
from lab11.src.trainer import Trainer


class ExperimentRunner:
    def __init__(self, num_samples_list, results_file='results/experiment_results.csv',
                 haralick_params_file='data/haralick_parameters.csv'):
        self.num_samples_list = num_samples_list
        self.results_file = results_file
        self.haralick_params_file = haralick_params_file  # Новый параметр
        os.makedirs(os.path.dirname(self.results_file), exist_ok=True)

    def run_experiments(self):
        # Загрузка параметров Харалика
        from lab11.src.haralick_data_loader import HaralickDataLoader
        haralick_params = HaralickDataLoader.get_haralick_params(self.haralick_params_file)

        fieldnames = ['num_train_samples', 'epochs', 'batch_size', 'train_time_sec', 'test_loss', 'test_accuracy']
        with open(self.results_file, 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            for num_train_samples in self.num_samples_list:
                print(f"Запуск эксперимента с {num_train_samples} обучающими примерами.")
                data_gen = DataGenerator(num_samples_per_disease=num_train_samples // 8,
                                         haralick_params=haralick_params)
                (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_gen.generate_all_datasets(
                    num_train_samples)

                # Нормализация данных
                X_train /= 1.0
                X_val /= 1.0
                X_test /= 1.0

                # Построение и компиляция модели
                model_builder = ModelBuilder()
                model = model_builder.build_model()
                trainer = Trainer(model)
                trainer.compile_model()

                # Обучение модели
                history, training_time = trainer.train(X_train, y_train, X_val, y_val)

                # Оценка модели
                evaluator = Evaluator(model)
                test_loss, test_accuracy = evaluator.evaluate(X_test, y_test)

                # Запись результатов
                writer.writerow({
                    'num_train_samples': num_train_samples,
                    'epochs': 200,
                    'batch_size': 20,
                    'train_time_sec': training_time,
                    'test_loss': test_loss,
                    'test_accuracy': test_accuracy
                })
                csv_file.flush()
                print(f"Эксперимент с {num_train_samples} обучающими примерами завершён.")
