import csv
import os
from datetime import datetime

from lab11.src.data_generator import DataGenerator
from lab11.src.evaluator import Evaluator
from lab11.src.haralick_data_loader import HaralickDataLoader
from lab11.src.model_builder import ModelBuilder
from lab11.src.trainer import Trainer


class ExperimentRunner:
    def __init__(self, num_samples_list, results_file='results/experiment_results.csv',
                 haralick_params_file='data/haralick_parameters.csv', stats_dir='results/per_run_stats',
                 model_types=None):
        if model_types is None:
            model_types = ['FFNN', 'CNN']
        self.num_samples_list = num_samples_list
        self.results_file = results_file
        self.haralick_params_file = haralick_params_file
        self.stats_dir = stats_dir
        self.model_types = model_types

        os.makedirs(os.path.dirname(self.results_file), exist_ok=True)
        os.makedirs(self.stats_dir, exist_ok=True)

    def run_experiments(self):
        # Загрузка параметров Харалика
        haralick_params = HaralickDataLoader.get_haralick_params(self.haralick_params_file)
        color_to_params = HaralickDataLoader.group_params_by_color(haralick_params)
        color_combinations = HaralickDataLoader.get_color_combinations(color_to_params)

        # Столбцы для результатов
        fieldnames = ['model_type', 'colors_used', 'num_train_samples', 'epochs', 'batch_size',
                      'train_time_sec', 'test_loss', 'test_accuracy']
        with open(self.results_file, 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            for model_type in self.model_types:
                for color_combination in color_combinations:
                    for num_train_samples in self.num_samples_list:
                        print(
                            f"Запуск эксперимента: {model_type}, цветов: {color_combination}, {num_train_samples} образцов.")

                        # Формирование уникального run_id
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        run_id = f"{model_type}_{'_'.join(color_combination)}_{num_train_samples}_{timestamp}"

                        filtered_params = HaralickDataLoader.get_params_for_combination(
                            color_combination, color_to_params, haralick_params)
                        data_gen = DataGenerator(num_samples_per_disease=num_train_samples // 8,
                                                 haralick_params=filtered_params)

                        (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_gen.generate_all_datasets(
                            num_train_samples)

                        # Получение формы данных
                        input_shape = X_train.shape[1:]  # (height, width, channels)
                        print(f"Используемая форма входных данных для модели {run_id}")

                        # Построение модели
                        model_builder = ModelBuilder(input_shape=input_shape, model_type=model_type)
                        model = model_builder.build_model()

                        # Обучение модели
                        trainer = Trainer(model)
                        trainer.compile_model()
                        history, train_time = trainer.train(X_train, y_train, X_val, y_val, run_id,
                                                            stats_dir=self.stats_dir)

                        # Оценка модели
                        evaluator = Evaluator(model)
                        test_loss, test_accuracy = evaluator.evaluate(X_test, y_test)

                        # Сохранение результатов
                        writer.writerow({
                            'model_type': model_type,
                            'colors_used': ','.join(color_combination),
                            'num_train_samples': num_train_samples,
                            'epochs': 200,
                            'batch_size': 20,
                            'train_time_sec': train_time,
                            'test_loss': test_loss,
                            'test_accuracy': test_accuracy
                        })
                        csv_file.flush()
                        print(
                            f"Эксперимент завершён: {model_type}, цветов: {color_combination}, {num_train_samples} образцов. "
                            f"run_id: {run_id}")
