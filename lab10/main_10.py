# main.py

from lab10.src.data_loader import DataLoader
from lab10.src.evaluator import Evaluator
from lab10.src.hyperparameter_tuner import HyperparameterTuner
from lab10.src.model_builder import ModelBuilder
from lab10.src.trainer import Trainer
from lab10.src.visualizer import Visualizer


def main():
    # Пути к директориям с данными
    train_dir = 'data/train_dir'
    val_dir = 'data/val_dir'
    test_dir = 'data/test_dir'

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
    visualizer = Visualizer(results_file='results/hyperparameter_results.csv')

    # Запуск подбора гиперпараметров
    tuner.tune(hyperparameters)

    # Построение графиков
    visualizer.generate_all_plots()
    print("Подбор гиперпараметров и построение графиков завершены.")


if __name__ == '__main__':
    main()
