# visualizer.py

import os

import matplotlib.pyplot as plt
import pandas as pd


class Visualizer:
    def __init__(self, results_file='results/experiment_results.csv', plots_dir='results/plots'):
        self.results_file = results_file
        self.plots_dir = plots_dir
        os.makedirs(self.plots_dir, exist_ok=True)

    def plot_results(self):
        df = pd.read_csv(self.results_file)

        # График точности
        plt.figure()
        plt.plot(df['num_train_samples'], df['test_accuracy'], marker='o')
        plt.title('Точность на тестовых данных vs Количество обучающих примеров')
        plt.xlabel('Количество обучающих примеров')
        plt.ylabel('Точность на тестовых данных')
        plt.grid(True)
        plt.savefig(os.path.join(self.plots_dir, 'test_accuracy_vs_num_samples.png'))
        plt.close()

        # График времени обучения
        plt.figure()
        plt.plot(df['num_train_samples'], df['train_time_sec'], marker='o')
        plt.title('Время обучения vs Количество обучающих примеров')
        plt.xlabel('Количество обучающих примеров')
        plt.ylabel('Время обучения (сек)')
        plt.grid(True)
        plt.savefig(os.path.join(self.plots_dir, 'training_time_vs_num_samples.png'))
        plt.close()


