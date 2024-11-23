import os

from lab11.src.experiment_runner import ExperimentRunner
from lab11.src.visualizer import Visualizer


def main():
    # Пути к директориям и файлам
    base_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, 'data')
    results_file = os.path.join(base_dir, 'results', 'experiment_results.csv')
    plots_dir = os.path.join(base_dir, 'results', 'plots')
    haralick_params_file = os.path.join(data_dir, 'haralick_parameters.csv')

    # Список количества обучающих примеров
    num_samples_list = [300, 600, 900, 1500]

    # Запуск экспериментов
    experiment_runner = ExperimentRunner(num_samples_list, results_file=results_file,
                                         haralick_params_file=haralick_params_file)
    experiment_runner.run_experiments()

    # Построение графиков
    visualizer = Visualizer(results_file=results_file, plots_dir=plots_dir)
    visualizer.plot_results()
    print("Эксперименты завершены и результаты сохранены.")


if __name__ == '__main__':
    main()
