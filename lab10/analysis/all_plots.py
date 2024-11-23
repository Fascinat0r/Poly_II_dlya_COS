import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from numpy.polynomial.polynomial import Polynomial

def plot_with_trendlines(data_folder, output_folder='plots'):
    """
    Построение графиков точности и потерь на тестовой выборке с линиями тренда.

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
        ax_acc.plot(epochs, val_accuracy, alpha=0.5, label=os.path.basename(run_file).split('.')[0])
        ax_loss.plot(epochs, val_loss, alpha=0.5, label=os.path.basename(run_file).split('.')[0])

    # Построение линий тренда для точности
    all_epochs = np.concatenate([pd.read_csv(f)['epoch'].values for f in run_files])
    all_val_accuracy = np.concatenate([pd.read_csv(f)['val_accuracy'].values for f in run_files])

    for order in [1, 2, 3]:
        coefs = np.polyfit(all_epochs, all_val_accuracy, order)
        trendline = np.polyval(coefs, np.linspace(all_epochs.min(), all_epochs.max(), 500))
        ax_acc.plot(np.linspace(all_epochs.min(), all_epochs.max(), 500), trendline, label=f'Тренд {order}-го порядка', linewidth=2)

    # Построение линий тренда для потерь
    all_val_loss = np.concatenate([pd.read_csv(f)['val_loss'].values for f in run_files])

    for order in [1, 2, 3]:
        coefs = np.polyfit(all_epochs, all_val_loss, order)
        trendline = np.polyval(coefs, np.linspace(all_epochs.min(), all_epochs.max(), 500))
        ax_loss.plot(np.linspace(all_epochs.min(), all_epochs.max(), 500), trendline, label=f'Тренд {order}-го порядка', linewidth=2)

    # Настройка графиков
    ax_acc.set_title('График точности на тестовой выборке')
    ax_acc.set_xlabel('Эпохи')
    ax_acc.set_ylabel('Точность')

    ax_loss.set_title('График потерь на тестовой выборке')
    ax_loss.set_xlabel('Эпохи')
    ax_loss.set_ylabel('Потери')

    # Сохранение графиков
    acc_output_path = os.path.join(output_folder, 'test_accuracy_plot.png')
    loss_output_path = os.path.join(output_folder, 'test_loss_plot.png')
    fig_acc.savefig(acc_output_path)
    fig_loss.savefig(loss_output_path)

    plt.show()
    print(f"Графики сохранены в {output_folder}:\n - {acc_output_path}\n - {loss_output_path}")

if __name__ == "__main__":
    # Путь к папке с файлами прогонов
    data_folder = '../results/per_run_stats'
    # Папка для сохранения графиков
    output_folder = '../results/plots'

    # Построение графиков
    plot_with_trendlines(data_folder, output_folder)
