# src/all_plots.py

import os
from glob import glob

import matplotlib.pyplot as plt
import pandas as pd


def plot_with_trendlines(data_folder, output_folder='plots'):
    """
    Построение графиков точности и потерь на тестовой выборке.

    :param data_folder: Путь к папке с файлами прогонов (../results/per_run_stats).
    :param output_folder: Папка для сохранения графиков.
    """
    # Убедимся, что папка для сохранения существует
    os.makedirs(output_folder, exist_ok=True)

    # Получаем список всех файлов
    run_files = glob(os.path.join(data_folder, "run_*.csv"))
    if not run_files:
        print(f"Файлы в папке {data_folder} не найдены.")
        return

    # Создаём фигуры для точности и потерь
    fig_acc, ax_acc = plt.subplots(figsize=(12, 8))
    fig_loss, ax_loss = plt.subplots(figsize=(12, 8))

    # Для каждой прогонки
    for run_file in run_files:
        # Загружаем данные
        run_data = pd.read_csv(run_file)
        epochs = run_data['epoch']
        val_accuracy = run_data['val_accuracy']
        val_loss = run_data['val_loss']

        # Отображаем графики точности и потерь
        label = os.path.basename(run_file).replace('.csv', '')
        ax_acc.plot(epochs, val_accuracy, alpha=0.5, label=label)
        ax_loss.plot(epochs, val_loss, alpha=0.5, label=label)

    # Настройка графиков
    ax_acc.set_title('График точности на тестовой выборке')
    ax_acc.set_xlabel('Эпохи')
    ax_acc.set_ylabel('Точность')
    ax_acc.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_acc.grid(True)

    ax_loss.set_title('График потерь на тестовой выборке')
    ax_loss.set_xlabel('Эпохи')
    ax_loss.set_ylabel('Потери')
    ax_loss.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_loss.grid(True)

    # Сохранение графиков
    acc_output_path = os.path.join(output_folder, 'test_accuracy_all_runs.png')
    loss_output_path = os.path.join(output_folder, 'test_loss_all_runs.png')
    fig_acc.savefig(acc_output_path, bbox_inches='tight')
    fig_loss.savefig(loss_output_path, bbox_inches='tight')

    plt.close(fig_acc)
    plt.close(fig_loss)
    print(f"Графики сохранены в {output_folder}:\n - {acc_output_path}\n - {loss_output_path}")


if __name__ == "__main__":
    # Путь к папке с файлами прогонов
    data_folder = 'results/per_run_stats'
    # Папка для сохранения графиков
    output_folder = 'results/plots'

    # Построение графиков
    plot_with_trendlines(data_folder, output_folder)
