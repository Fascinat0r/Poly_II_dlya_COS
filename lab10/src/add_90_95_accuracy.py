import os

import pandas as pd


def update_results_with_epochs(results_file, stats_dir):
    # Читаем основной файл с результатами
    results_df = pd.read_csv(results_file)

    # Столбцы для новых данных
    results_df['epoch_90_accuracy'] = None
    results_df['epoch_95_accuracy'] = None

    # Идем по каждой строке (прогонке)
    for index, row in results_df.iterrows():
        run_id = f"run_{index + 1}.csv"
        run_file_path = os.path.join(stats_dir, run_id)

        # Проверяем, существует ли файл статистики для данной прогонки
        if os.path.exists(run_file_path):
            # Читаем данные эпох из файла
            epoch_stats = pd.read_csv(run_file_path)

            # Найти первую эпоху, где точность >= 90%
            epoch_90 = epoch_stats.loc[epoch_stats['val_accuracy'] >= 0.90, 'epoch'].min()
            # Найти первую эпоху, где точность >= 95%
            epoch_95 = epoch_stats.loc[epoch_stats['val_accuracy'] >= 0.95, 'epoch'].min()

            # Добавляем данные в соответствующие столбцы
            results_df.at[index, 'epoch_90_accuracy'] = epoch_90 if pd.notna(epoch_90) else None
            results_df.at[index, 'epoch_95_accuracy'] = epoch_95 if pd.notna(epoch_95) else None
        else:
            print(f"Файл статистики для {run_id} не найден. Пропускаем.")

    # Сохраняем обновленные результаты в тот же файл
    results_df.to_csv(results_file, index=False)
    print(f"Результаты успешно обновлены. Добавлены столбцы epoch_90_accuracy и epoch_95_accuracy.")


# Путь к файлам
if __name__ == "__main__":
    results_file = '../results/hyperparameter_results.csv'
    stats_dir = '../results/per_run_stats'

    # Запуск обновления
    update_results_with_epochs(results_file, stats_dir)
