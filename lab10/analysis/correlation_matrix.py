import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


def build_filtered_correlation_matrix(input_file, output_image='filtered_correlation_matrix.png'):
    """
    Построение матрицы корреляции на основе фильтрованных данных из CSV-файла.
    Выводит отсортированные корреляции в консоль и сохраняет тепловую карту.

    :param input_file: Путь к входному CSV-файлу.
    :param output_image: Путь для сохранения изображения матрицы корреляции.
    """
    # Проверяем, существует ли входной файл
    if not os.path.exists(input_file):
        print(f"Файл {input_file} не найден.")
        return

    # Загружаем данные из CSV
    df = pd.read_csv(input_file)

    # Выбираем только гиперпараметры и ключевые результаты
    filtered_columns = [
        'num_hidden_layers', 'neurons_per_layer', 'epochs', 'batch_size',  # Гиперпараметры
        'test_accuracy', 'training_time_sec',  # Конечные результаты
        'epoch_90_accuracy', 'epoch_95_accuracy'  # Эпохи до 90% и 95% точности
    ]
    filtered_df = df[filtered_columns]

    # Строим матрицу корреляции
    correlation_matrix = filtered_df.corr()

    # Визуализация с помощью тепловой карты
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, center=0)
    plt.title('Матрица корреляции (гиперпараметры и ключевые результаты)')
    plt.tight_layout()

    # Сохранение изображения
    plt.savefig(output_image)
    plt.show()

    # Вывод корреляций в консоль
    print("\nКорреляции (отсортированы по модулю):")
    correlation_pairs = correlation_matrix.unstack()  # Разворачиваем матрицу в пары
    sorted_correlation = correlation_pairs.sort_values(key=lambda x: abs(x), ascending=False)

    # Убираем повторения (оставляем уникальные пары)
    seen = set()
    for (col1, col2), value in sorted_correlation.items():
        if col1 == col2 or (col2, col1) in seen:  # Пропускаем диагональ и дублирующиеся пары
            continue
        seen.add((col1, col2))
        print(f"{col1} <-> {col2}: {value:.3f}")

    print(f"\nМатрица корреляции сохранена в {output_image}")


if __name__ == "__main__":
    # Путь к CSV-файлу с данными
    input_file = '../results/hyperparameter_results.csv'
    # Путь для сохранения изображения
    output_image = '../results/filtered_correlation_matrix.png'

    # Построение матрицы корреляции
    build_filtered_correlation_matrix(input_file, output_image)
