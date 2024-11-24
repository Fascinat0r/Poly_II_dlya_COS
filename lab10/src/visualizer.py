# src/visualizer.py

import os

import matplotlib.pyplot as plt
import pandas as pd


class Visualizer:
    def __init__(self, results_file='results/hyperparameter_results.csv', plots_dir='results/plots'):
        self.results_file = results_file
        self.plots_dir = plots_dir
        os.makedirs(self.plots_dir, exist_ok=True)

    def plot_epochs_vs_accuracy(self):
        df = pd.read_csv(self.results_file)
        grouped = df.groupby('epochs')['test_accuracy'].mean().reset_index()
        plt.figure()
        plt.plot(grouped['epochs'], grouped['test_accuracy'], marker='o')
        plt.title('Количество эпох обучения vs Точность на тестовых данных')
        plt.xlabel('Количество эпох')
        plt.ylabel('Точность на тестовых данных')
        plt.grid(True)
        plt.savefig(os.path.join(self.plots_dir, 'epochs_vs_accuracy.png'))
        plt.close()

    def plot_batch_size_vs_accuracy(self):
        df = pd.read_csv(self.results_file)
        grouped = df.groupby('batch_size')['test_accuracy'].mean().reset_index()
        plt.figure()
        plt.plot(grouped['batch_size'], grouped['test_accuracy'], marker='o')
        plt.title('Размер мини-выборки vs Точность на тестовых данных')
        plt.xlabel('Размер мини-выборки')
        plt.ylabel('Точность на тестовых данных')
        plt.grid(True)
        plt.savefig(os.path.join(self.plots_dir, 'batch_size_vs_accuracy.png'))
        plt.close()

    def plot_neurons_vs_accuracy(self):
        df = pd.read_csv(self.results_file)
        grouped = df.groupby('neurons_per_layer')['test_accuracy'].mean().reset_index()
        plt.figure()
        plt.plot(grouped['neurons_per_layer'], grouped['test_accuracy'], marker='o')
        plt.title('Количество нейронов в скрытых слоях vs Точность на тестовых данных')
        plt.xlabel('Количество нейронов в скрытых слоях')
        plt.ylabel('Точность на тестовых данных')
        plt.grid(True)
        plt.savefig(os.path.join(self.plots_dir, 'neurons_vs_accuracy.png'))
        plt.close()

    def plot_layers_vs_accuracy(self):
        df = pd.read_csv(self.results_file)
        grouped = df.groupby('num_hidden_layers')['test_accuracy'].mean().reset_index()
        plt.figure()
        plt.plot(grouped['num_hidden_layers'], grouped['test_accuracy'], marker='o')
        plt.title('Количество скрытых слоев vs Точность на тестовых данных')
        plt.xlabel('Количество скрытых слоев')
        plt.ylabel('Точность на тестовых данных')
        plt.grid(True)
        plt.savefig(os.path.join(self.plots_dir, 'layers_vs_accuracy.png'))
        plt.close()

    def generate_all_plots(self):
        self.plot_epochs_vs_accuracy()
        self.plot_batch_size_vs_accuracy()
        self.plot_neurons_vs_accuracy()
        self.plot_layers_vs_accuracy()
