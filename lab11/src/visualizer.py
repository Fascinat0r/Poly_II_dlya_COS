# src/visualizer.py

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

        # Получить список моделей и цветовых компонентов
        model_types = df['model_type'].unique()
        color_components = df['colors_used'].unique()

        # Создать фигуры для точности, потерь и эпох достижения точности
        fig_acc, ax_acc = plt.subplots(figsize=(12, 8))
        fig_loss, ax_loss = plt.subplots(figsize=(12, 8))
        fig_epochs, ax_epochs = plt.subplots(figsize=(12, 8))

        for model_type in model_types:
            for components in color_components:
                subset = df[(df['model_type'] == model_type) & (df['colors_used'] == components)]
                label = f"{model_type} ({components})"
                ax_acc.plot(subset['num_train_samples'], subset['test_accuracy'], marker='o', label=label)
                ax_loss.plot(subset['num_train_samples'], subset['test_loss'], marker='o', label=label)
                ax_epochs.plot(subset['num_train_samples'], subset['epoch_90_accuracy'], marker='x', linestyle='--',
                               label=f"{label} - 90%")
                ax_epochs.plot(subset['num_train_samples'], subset['epoch_95_accuracy'], marker='s', linestyle=':',
                               label=f"{label} - 95%")

        # Настройка графиков точности
        ax_acc.set_title('Точность на тестовых данных vs Количество обучающих примеров')
        ax_acc.set_xlabel('Количество обучающих примеров')
        ax_acc.set_ylabel('Точность на тестовых данных')
        ax_acc.grid(True)
        ax_acc.legend()

        # Настройка графиков потерь
        ax_loss.set_title('Потери на тестовых данных vs Количество обучающих примеров')
        ax_loss.set_xlabel('Количество обучающих примеров')
        ax_loss.set_ylabel('Потери на тестовых данных')
        ax_loss.grid(True)
        ax_loss.legend()

        # Настройка графиков эпох достижения точности
        ax_epochs.set_title('Эпохи достижения 90% и 95% точности')
        ax_epochs.set_xlabel('Количество обучающих примеров')
        ax_epochs.set_ylabel('Эпохи')
        ax_epochs.grid(True)
        ax_epochs.legend()

        # Сохранение графиков
        acc_output_path = os.path.join(self.plots_dir, 'test_accuracy_vs_num_samples.png')
        loss_output_path = os.path.join(self.plots_dir, 'test_loss_vs_num_samples.png')
        epochs_output_path = os.path.join(self.plots_dir, 'epochs_accuracy_reached.png')

        fig_acc.savefig(acc_output_path)
        fig_loss.savefig(loss_output_path)
        fig_epochs.savefig(epochs_output_path)

        plt.close(fig_acc)
        plt.close(fig_loss)
        plt.close(fig_epochs)
        print(
            f"Графики сохранены в {self.plots_dir}:\n - {acc_output_path}\n - {loss_output_path}\n - {epochs_output_path}")
