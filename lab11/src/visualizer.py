import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class Visualizer:
    def __init__(self, results_file='results/experiment_results.csv', plots_dir='results/plots'):
        self.results_file = results_file
        self.plots_dir = plots_dir
        os.makedirs(self.plots_dir, exist_ok=True)

    def plot_train_time_vs_samples(self, df):
        plt.figure(figsize=(12, 8))
        sns.lineplot(data=df, x='num_train_samples', y='train_time_sec', hue='model_type', style='colors_used',
                     markers=True, dashes=False)
        plt.title('Время обучения vs Количество обучающих примеров')
        plt.xlabel('Количество обучающих примеров')
        plt.ylabel('Время обучения (секунды)')
        plt.legend(title='Тип модели / Цвета', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        output_path = os.path.join(self.plots_dir, 'train_time_vs_num_samples.png')
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        print(f"График времени обучения сохранён: {output_path}")

        # Вывод результатов для этого графика
        print("\nРезультаты для графика 'Время обучения vs Количество обучающих примеров':")
        print(df[['model_type', 'colors_used', 'num_train_samples', 'train_time_sec']]
              .sort_values(by='train_time_sec').to_string(index=False))

    def plot_correlation_matrix(self, df):
        plt.figure(figsize=(12, 10))

        df['num_haralik_params'] = df['colors_used'].apply(lambda x: len(x))
        df_encoded = pd.get_dummies(df, columns=['model_type'], drop_first=True)

        numeric_cols = ['num_train_samples', 'train_time_sec', 'test_loss', 'test_accuracy', 'epoch_90_accuracy',
                        'epoch_95_accuracy', 'num_haralik_params']

        corr = df_encoded[numeric_cols].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title('Корреляционная матрица')

        output_path = os.path.join(self.plots_dir, 'correlation_matrix.png')
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

        print(f"Корреляционная матрица сохранена: {output_path}")

        # Вывод результатов для этого графика
        print("\nРезультаты для графика 'Корреляционная матрица':")
        corr_pairs = corr.stack().reset_index()
        corr_pairs.columns = ['Parameter 1', 'Parameter 2', 'Correlation']
        corr_pairs = corr_pairs[corr_pairs['Parameter 1'] < corr_pairs['Parameter 2']]
        corr_pairs['Abs Correlation'] = corr_pairs['Correlation'].abs()
        print(corr_pairs.sort_values(by='Abs Correlation', ascending=False).to_string(index=False))

    def plot_boxplots(self, df):
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='model_type', y='test_accuracy', hue='colors_used', data=df)
        plt.title('Распределение точности по типу модели и цветовым компонентам')
        plt.xlabel('Тип модели')
        plt.ylabel('Точность на тестовых данных')
        plt.legend(title='Цвета', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        output_path = os.path.join(self.plots_dir, 'boxplot_test_accuracy.png')
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        print(f"Boxplot точности сохранён: {output_path}")

        print("\nРезультаты для графика 'Распределение точности по типу модели и цветовым компонентам':")
        print(df[['model_type', 'colors_used', 'test_accuracy']].sort_values(by='test_accuracy',
                                                                             ascending=False).to_string(index=False))

        plt.figure(figsize=(12, 8))
        sns.boxplot(x='model_type', y='test_loss', hue='colors_used', data=df)
        plt.title('Распределение потерь по типу модели и цветовым компонентам')
        plt.xlabel('Тип модели')
        plt.ylabel('Потери на тестовых данных')
        plt.legend(title='Цвета', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        output_path = os.path.join(self.plots_dir, 'boxplot_test_loss.png')
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        print(f"Boxplot потерь сохранён: {output_path}")

        print("\nРезультаты для графика 'Распределение потерь по типу модели и цветовым компонентам':")
        print(df[['model_type', 'colors_used', 'test_loss']].sort_values(by='test_loss').to_string(index=False))

        plt.figure(figsize=(12, 8))
        sns.boxplot(x='model_type', y='train_time_sec', hue='colors_used', data=df)
        plt.title('Распределение времени обучения по типу модели и цветовым компонентам')
        plt.xlabel('Тип модели')
        plt.ylabel('Время обучения (секунды)')
        plt.legend(title='Цвета', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        output_path = os.path.join(self.plots_dir, 'boxplot_train_time.png')
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        print(f"Boxplot времени обучения сохранён: {output_path}")

        print("\nРезультаты для графика 'Распределение времени обучения по типу модели и цветовым компонентам':")
        print(
            df[['model_type', 'colors_used', 'train_time_sec']].sort_values(by='train_time_sec').to_string(index=False))

    def plot_all(self):
        df = pd.read_csv(self.results_file)

        if df['colors_used'].dtype != object:
            df['colors_used'] = df['colors_used'].astype(str)

        self.plot_train_time_vs_samples(df)
        self.plot_correlation_matrix(df)
        self.plot_boxplots(df)


if __name__ == "__main__":
    results_file = '../results/experiment_results.csv'
    plots_dir = '../results/plots'

    visualizer = Visualizer(results_file=results_file, plots_dir=plots_dir)
    visualizer.plot_all()
