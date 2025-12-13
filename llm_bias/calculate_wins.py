import csv
import os
from pathlib import Path

# Маппинг имен моделей из названия файла в имена в CSV
MODEL_NAME_MAPPING = {
    'deepseek': 'DeepSeek',
    'flash': 'Flash',
    'gemini': 'Gemini',
    'qwen': 'Qwen'
}

# Индексы колонок: task_id=0, task_type=1, model_name=2, evaluator=3, критерии начинаются с 4
CRITERIA_START_INDEX = 4
NUM_CRITERIA = 8

def get_model_name_from_filename(filename):
    """Извлекает имя модели из названия файла"""
    base_name = Path(filename).stem
    model_name_lower = base_name.split('_')[0].lower()
    return MODEL_NAME_MAPPING.get(model_name_lower, model_name_lower.capitalize())

def process_file(filepath):
    """Обрабатывает один CSV файл"""
    filename = os.path.basename(filepath)
    model_name = get_model_name_from_filename(filename)
    
    wins = 0
    unique_questions = set()
    wins_by_criterion = {}  # Словарь для подсчета побед по каждому критерию
    
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader)  # Читаем заголовок
        
        # Получаем названия критериев из заголовка
        criteria_names = header[CRITERIA_START_INDEX:CRITERIA_START_INDEX + NUM_CRITERIA]
        
        # Инициализируем счетчики для каждого критерия
        for criterion in criteria_names:
            wins_by_criterion[criterion] = 0
        
        for row in reader:
            if row[2] == model_name:  # model_name колонка
                # Добавляем уникальный вопрос (task_id + evaluator)
                unique_questions.add((row[0], row[3]))
                
                # Проверяем каждый критерий (колонки 4-11)
                for i, criterion in enumerate(criteria_names):
                    col_index = CRITERIA_START_INDEX + i
                    if col_index < len(row):
                        # Победа = оценка равна 1 (или 1.0)
                        try:
                            score = float(row[col_index])
                            if score == 1.0:
                                wins += 1
                                wins_by_criterion[criterion] += 1
                        except (ValueError, IndexError):
                            continue
    
    total_comparisons = len(unique_questions) * NUM_CRITERIA
    win_percentage = (wins / total_comparisons * 100) if total_comparisons > 0 else 0
    
    # Вычисляем процент побед по каждому критерию
    wins_percentage_by_criterion = {}
    for criterion, criterion_wins in wins_by_criterion.items():
        criterion_total = len(unique_questions)
        wins_percentage_by_criterion[criterion] = (criterion_wins / criterion_total * 100) if criterion_total > 0 else 0
    
    return {
        'filename': filename,
        'model_name': model_name,
        'wins': wins,
        'total_comparisons': total_comparisons,
        'unique_questions': len(unique_questions),
        'win_percentage': win_percentage,
        'wins_by_criterion': wins_by_criterion,
        'wins_percentage_by_criterion': wins_percentage_by_criterion
    }

def main():
    # Путь к директории с CSV файлами
    directory = Path(__file__).parent
    
    # Находим все CSV файлы
    csv_files = [f for f in directory.glob('llm_self_rankings/*.csv') 
                 if f.name != 'calculate_wins.py' and f.name != 'win_statistics.csv']
    
    if not csv_files:
        print("Не найдено CSV файлов в директории")
        return
    
    results = []
    
    # Обрабатываем каждый файл
    for csv_file in sorted(csv_files):
        try:
            result = process_file(csv_file)
            results.append(result)
        except Exception as e:
            print(f"Ошибка при обработке {csv_file.name}: {e}")
    
    # Выводим результаты
    print("\n" + "="*80)
    print("РЕЗУЛЬТАТЫ АНАЛИЗА ПОБЕД МОДЕЛЕЙ")
    print("="*80 + "\n")
    
    for result in results:
        print(f"Файл: {result['filename']}")
        print(f"  Модель: {result['model_name']}")
        print(f"  Количество побед: {result['wins']}")
        print(f"  Всего сравнений: {result['total_comparisons']}")
        print(f"  Уникальных вопросов: {result['unique_questions']}")
        print(f"  Средний процент побед: {result['win_percentage']:.2f}%")
        print()
    
    # Выводим статистику по категориям
    print("="*80)
    print("СТАТИСТИКА ПО КАТЕГОРИЯМ")
    print("="*80 + "\n")
    
    for result in results:
        print(f"Файл: {result['filename']} ({result['model_name']})")
        print("-" * 80)
        
        # Сортируем критерии по количеству побед (по убыванию)
        sorted_criteria = sorted(
            result['wins_by_criterion'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for criterion, wins_count in sorted_criteria:
            percentage = result['wins_percentage_by_criterion'][criterion]
            print(f"  {criterion:25s}: {wins_count:3d} побед ({percentage:5.2f}%)")
        print()
    
    # Сохраняем результаты в CSV
    output_file = directory / 'win_statistics.csv'
    
    # Подготовка данных для CSV (включая категории)
    csv_data = []
    for result in results:
        row = {
            'filename': result['filename'],
            'model_name': result['model_name'],
            'wins': result['wins'],
            'total_comparisons': result['total_comparisons'],
            'unique_questions': result['unique_questions'],
            'win_percentage': result['win_percentage']
        }
        # Добавляем статистику по каждой категории
        for criterion, wins_count in result['wins_by_criterion'].items():
            row[f'{criterion}_wins'] = wins_count
            row[f'{criterion}_percentage'] = result['wins_percentage_by_criterion'][criterion]
        csv_data.append(row)
    
    # Получаем все возможные поля для CSV
    fieldnames = ['filename', 'model_name', 'wins', 'total_comparisons', 
                  'unique_questions', 'win_percentage']
    if results:
        # Добавляем поля для категорий из первого результата
        for criterion in results[0]['wins_by_criterion'].keys():
            fieldnames.append(f'{criterion}_wins')
            fieldnames.append(f'{criterion}_percentage')
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)
    
    print(f"Результаты сохранены в {output_file}")

if __name__ == '__main__':
    main()
