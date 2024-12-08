# main.py
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from lab10.src.add_90_95_accuracy import update_results_with_epochs
from lab10.src.data_loader import DataLoader
from lab10.src.evaluator import Evaluator
from lab10.src.hyperparameter_tuner import HyperparameterTuner
from lab10.src.model_builder import ModelBuilder
from lab10.src.trainer import Trainer
from lab10.src.utils import clear_data
from lab10.src.visualizer import Visualizer


def main():
    # Пути к директориям с данными
    base_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, 'data')
    results_dir = os.path.join(base_dir, 'results')
    train_dir = os.path.join(data_dir, 'train_dir')
    val_dir = os.path.join(data_dir, 'val_dir')
    test_dir = os.path.join(data_dir, 'test_dir')
    results_file = os.path.join(results_dir, 'hyperparameter_results.csv')
    stats_dir = os.path.join(results_dir, 'per_run_stats')
    plots_dir = os.path.join(results_dir, 'plots')

    # Предложить очистку данных
    if os.path.exists(results_file) or os.path.exists(stats_dir) or os.path.exists(plots_dir):
        clear_choice = input("Очистить существующие данные (y/N)? ").strip().lower()
        if clear_choice == 'y':
            clear_data(results_file, stats_dir, plots_dir)

    # Гиперпараметры для подбора
    hyperparameters = {
        'num_hidden_layers': [1, 2, 3],  # Количество скрытых слоев
        'neurons_per_layer': [500, 700, 900, 1200],  # Количество нейронов в скрытых слоях
        'epochs': [50, 75, 100, 125],  # Количество эпох обучения
        'batch_size': [20, 50, 100]  # Размер мини-выборки
    }

    # Инициализация компонентов
    data_loader = DataLoader(train_dir, val_dir, test_dir, img_width=28, img_height=28)
    model_builder = ModelBuilder(input_shape=(28, 28, 3), num_classes=9)
    trainer = Trainer(model=None)  # Модель будет задана в процессе
    evaluator = Evaluator(model=None)  # Модель будет задана в процессе
    tuner = HyperparameterTuner(model_builder, data_loader, trainer, evaluator,
                                results_file='results/hyperparameter_results.csv', stats_dir='results/per_run_stats')
    visualizer = Visualizer(results_file, stats_dir, plots_dir)

    # Запуск подбора гиперпараметров
    tuner.tune(hyperparameters)

    # Добавляем данные об эпохах 90% и 95% точности
    update_results_with_epochs(results_file, stats_dir)

    # Построение графиков
    visualizer.generate_all_plots()
    print("Подбор гиперпараметров и построение графиков завершены.")


if __name__ == '__main__':
    main()
