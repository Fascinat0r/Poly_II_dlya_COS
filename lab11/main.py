# main.py
from src.experiment_runner import ExperimentRunner
from src.visualizer import Visualizer


def main():
    num_samples_list = [300, 600, 900, 1500]
    experiment_runner = ExperimentRunner(num_samples_list)
    experiment_runner.run_experiments()

    visualizer = Visualizer()
    visualizer.plot_results()
    print("Эксперименты завершены и результаты сохранены.")

if __name__ == '__main__':
    main()
