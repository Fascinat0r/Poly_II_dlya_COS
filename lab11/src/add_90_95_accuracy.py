import os
import pandas as pd


def update_results_with_epochs(results_file, stats_dir):
    # Читаем основной файл с результатами
    results_df = pd.read_csv(results_file)

    # Столбцы для новых данных
    if 'epoch_90_accuracy' not in results_df.columns:
        results_df['epoch_90_accuracy'] = None
    if 'epoch_95_accuracy' not in results_df.columns:
        results_df['epoch_95_accuracy'] = None

    # Получаем список всех файлов в stats_dir
    stat_files = [f for f in os.listdir(stats_dir) if f.endswith('.csv')]

    for stat_file in stat_files:
        try:
            # Извлекаем количество обучающих примеров из имени файла
            num_train_samples = int(stat_file.replace('run_', '').replace('.csv', ''))
            run_file_path = os.path.join(stats_dir, stat_file)

            # Проверяем, существует ли файл статистики
            if os.path.exists(run_file_path):
                # Читаем данные эпох из файла
                epoch_stats = pd.read_csv(run_file_path)

                # Найти первую эпоху, где точность >= 90%
                epoch_90 = epoch_stats.loc[epoch_stats['val_accuracy'] >= 0.90, 'epoch'].min()
                # Найти первую эпоху, где точность >= 95%
                epoch_95 = epoch_stats.loc[epoch_stats['val_accuracy'] >= 0.95, 'epoch'].min()

                # Обновляем строку в results_df для соответствующего количества обучающих примеров
                row_index = results_df[results_df['num_train_samples'] == num_train_samples].index
                if not row_index.empty:
                    results_df.at[row_index[0], 'epoch_90_accuracy'] = epoch_90 if pd.notna(epoch_90) else None
                    results_df.at[row_index[0], 'epoch_95_accuracy'] = epoch_95 if pd.notna(epoch_95) else None
                else:
                    print(f"Запись для {num_train_samples} не найдена в файле результатов. Пропускаем.")
            else:
                print(f"Файл {run_file_path} не найден. Пропускаем.")
        except ValueError:
            print(f"Не удалось извлечь количество обучающих примеров из имени файла: {stat_file}. Пропускаем.")
        except Exception as e:
            print(f"Ошибка при обработке файла {stat_file}: {e}")

    # Сохраняем обновленные результаты в тот же файл
    results_df.to_csv(results_file, index=False)
    print(f"Результаты успешно обновлены. Добавлены столбцы epoch_90_accuracy и epoch_95_accuracy.")


# Путь к файлам
if __name__ == "__main__":
    results_file = '../results/experiment_results.csv'  # Укажите путь к файлу с результатами
    stats_dir = '../results/per_run_stats'  # Укажите путь к папке с файлами статистики

    # Запуск обновления
    update_results_with_epochs(results_file, stats_dir)
