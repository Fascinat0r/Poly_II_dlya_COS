import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import shutil

from lab11.src.add_90_95_accuracy import update_results_with_epochs
from lab11.src.experiment_runner import ExperimentRunner
from lab11.src.visualizer import Visualizer


def clear_data(results_file, stats_dir, plots_dir):
    """
    Очистка результатов, прогонов и графиков.
    """
    if os.path.exists(results_file):
        os.remove(results_file)
    if os.path.exists(stats_dir):
        shutil.rmtree(stats_dir)
    os.makedirs(stats_dir, exist_ok=True)
    if os.path.exists(plots_dir):
        shutil.rmtree(plots_dir)
    os.makedirs(plots_dir, exist_ok=True)
    print("Все данные очищены.")


def main():
    # Пути к директориям и файлам
    base_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, 'data')
    results_file = os.path.join(base_dir, 'results', 'experiment_results.csv')
    plots_dir = os.path.join(base_dir, 'results', 'plots')
    stats_dir = os.path.join(base_dir, 'results', 'per_run_stats')
    haralick_params_file = os.path.join(data_dir, 'haralick_parameters.csv')

    # Предложить очистку данных
    if os.path.exists(results_file) or os.path.exists(stats_dir) or os.path.exists(plots_dir):
        clear_choice = input("Очистить существующие данные (y/N)? ").strip().lower()
        if clear_choice == 'y':
            clear_data(results_file, stats_dir, plots_dir)

    # Список количества обучающих примеров
    num_samples_list = [300, 600, 900, 1500]

    # Список типов моделей
    model_types = ['FFNN', 'CNN']

    # Запуск экспериментов
    experiment_runner = ExperimentRunner(
        num_samples_list=num_samples_list,
        results_file=results_file,
        haralick_params_file=haralick_params_file,
        stats_dir=stats_dir,
        model_types=model_types,
    )
    experiment_runner.run_experiments()

    # Обновление результатов с эпохами достижения точности
    update_results_with_epochs(results_file, stats_dir)

    # Построение графиков
    visualizer = Visualizer(results_file=results_file, plots_dir=plots_dir)
    visualizer.plot_results()
    print("Эксперименты завершены и результаты сохранены.")


if __name__ == '__main__':
    main()
