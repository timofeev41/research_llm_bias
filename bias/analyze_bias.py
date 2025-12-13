#!/usr/bin/env python3
"""
Скрипт для выявления смещения (bias) и неосмысленного оценивания
в результатах пользовательского опроса по качеству ответов LLM.

Метрики:
1. Model bias - средний балл каждой модели у конкретного респондента
2. Monotonicity score - проверка механического паттерна оценивания (Spearman correlation)
3. Variance score - дисперсия всех выставленных оценок
4. Verbosity bias - предпочтение длинных/коротких ответов
"""

import json
import re
from pathlib import Path
from typing import Any
import numpy as np
from scipy import stats
import pandas as pd


# Константы
MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash", 
    "DeepSeek-R1-0528",
    "Qwen3-235B-A22B-2507"
]

# Критерии оценки (только ranking вопросы)
RANKING_CRITERIA = [
    "fluency",
    "coherence", 
    "conciseness",
    "sentiment",
    "motivational",
    "constructiveness",
    "final"
]


def load_json(filepath: Path) -> dict:
    """Загрузка JSON файла."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_ranking_questions(responses: dict) -> list[tuple[str, list[str]]]:
    """
    Извлечение только ranking вопросов из ответов.
    Возвращает список кортежей (название_вопроса, ранжирование).
    """
    ranking_questions = []
    
    for key, value in responses.items():
        # Пропускаем sanity check и factual_accuracy (checkbox)
        if key.startswith('sanity_check'):
            continue
        if 'factual_accuracy' in key:
            continue
            
        # Проверяем, что это ranking вопрос (список из 4 моделей)
        if isinstance(value, list) and len(value) == 4:
            # Проверяем, что все элементы - названия моделей
            if all(model in MODELS for model in value):
                ranking_questions.append((key, value))
    
    return ranking_questions


def rank_to_score(ranking: list[str], model: str) -> int:
    """
    Конвертация позиции в ранге в балл.
    Позиция 0 (1-е место) = 4 балла
    Позиция 3 (4-е место) = 1 балл
    """
    if model in ranking:
        position = ranking.index(model)
        return 4 - position  # 4, 3, 2, 1
    return 0


def calculate_model_bias(ranking_questions: list[tuple[str, list[str]]]) -> dict[str, float]:
    """
    Расчёт среднего балла каждой модели у респондента.
    """
    model_scores = {model: [] for model in MODELS}
    
    for _, ranking in ranking_questions:
        for model in MODELS:
            score = rank_to_score(ranking, model)
            model_scores[model].append(score)
    
    # Средний балл по каждой модели
    return {
        model: np.mean(scores) if scores else 0.0 
        for model, scores in model_scores.items()
    }


def calculate_monotonicity_score(ranking_questions: list[tuple[str, list[str]]]) -> float:
    """
    Расчёт Monotonicity score через Spearman correlation.
    
    Идея: если респондент механически ставит оценки в одном порядке 
    (например, всегда первый вариант на 1 место, второй на 2 и т.д.),
    то корреляция между порядком представления и рангами будет высокой.
    
    Возвращает среднюю абсолютную корреляцию по всем вопросам.
    """
    if not ranking_questions:
        return 0.0
    
    correlations = []
    
    # Для каждого вопроса проверяем корреляцию порядка
    for _, ranking in ranking_questions:
        # Порядок представления (0, 1, 2, 3)
        presentation_order = list(range(len(ranking)))
        # Ранги (позиции моделей)
        ranks = list(range(len(ranking)))
        
        # Spearman correlation
        if len(set(ranking)) > 1:  # Нужна вариативность
            # Корреляция между индексами моделей в одном порядке и в другом
            corr, _ = stats.spearmanr(presentation_order, ranks)
            if not np.isnan(corr):
                correlations.append(abs(corr))
    
    # Альтернативный подход: проверяем консистентность порядка между вопросами
    # Сравниваем ранги моделей между последовательными вопросами
    if len(ranking_questions) >= 2:
        sequence_correlations = []
        for i in range(len(ranking_questions) - 1):
            _, ranking1 = ranking_questions[i]
            _, ranking2 = ranking_questions[i + 1]
            
            # Позиции моделей в первом и втором ранжировании
            positions1 = [ranking1.index(m) for m in MODELS]
            positions2 = [ranking2.index(m) for m in MODELS]
            
            corr, _ = stats.spearmanr(positions1, positions2)
            if not np.isnan(corr):
                sequence_correlations.append(corr)
        
        if sequence_correlations:
            # Высокое значение = модели всегда в одном порядке (bias)
            return float(np.mean(sequence_correlations))
    
    return 0.0


def calculate_variance_score(ranking_questions: list[tuple[str, list[str]]]) -> float:
    """
    Расчёт дисперсии всех выставленных оценок.
    
    Собираем все баллы для всех моделей и считаем дисперсию.
    Низкая дисперсия = оценки ставились без различий.
    """
    all_scores = []
    
    for _, ranking in ranking_questions:
        for model in MODELS:
            score = rank_to_score(ranking, model)
            all_scores.append(score)
    
    if not all_scores:
        return 0.0
    
    return float(np.var(all_scores))


def calculate_position_bias(ranking_questions: list[tuple[str, list[str]]]) -> dict[int, float]:
    """
    Дополнительная метрика: как часто модель с определённой позиции
    в форме попадает на 1-е место.
    
    Возвращает вероятность выбора для каждой позиции (0-3).
    """
    position_first_count = {0: 0, 1: 0, 2: 0, 3: 0}
    total = len(ranking_questions)
    
    if total == 0:
        return {i: 0.0 for i in range(4)}
    
    # Для этого нужны данные о порядке в форме, которые мы не имеем здесь
    # Пока возвращаем заглушку - эта метрика требует сопоставления с формами
    return {i: 0.25 for i in range(4)}


def extract_model_texts_from_html(html_content: str) -> dict[str, str]:
    """
    Извлечение текстов ответов моделей из HTML.
    
    Формат в HTML:
    <strong><span style='color:...;'>Animal Name</span></strong><br/>Text content
    
    Возвращает словарь {цвет_животного: текст}
    """
    # Паттерн для извлечения текстов моделей
    # Ищем: <span style='color:...;'>Name</span></strong><br/>Text
    pattern = r"<strong><span style='color:([^']+);'>([^<]+)</span></strong><br/>([^<]+(?:<[^>]+>[^<]*</[^>]+>)*)"
    
    # Более простой подход - разбиваем по маркерам моделей
    model_texts = {}
    
    # Цвета и их животные
    color_mapping = {
        '#04AFF2': 'Blue Whale',
        'red': 'Red Panda', 
        'goldenrod': 'Golden Fish',
        '#04F247': 'Green Elephant'
    }
    
    for color, animal in color_mapping.items():
        # Ищем текст после маркера животного
        pattern = rf"<span style='color:{re.escape(color)};?'>{re.escape(animal)}</span></strong><br/>([^<]+)"
        match = re.search(pattern, html_content)
        if match:
            model_texts[animal] = match.group(1).strip()
    
    return model_texts


def extract_task_model_lengths(form_data: dict) -> dict[str, dict[str, int]]:
    """
    Извлечение длин текстов моделей для каждой задачи из формы.
    
    Возвращает: {task_id: {model_name: text_length}}
    """
    task_lengths = {}
    
    for page in form_data.get('pages', []):
        page_name = page.get('name', '')
        
        # Пропускаем sanity check страницы
        if 'sanity_check' in page_name:
            continue
        
        for element in page.get('elements', []):
            if element.get('type') == 'html':
                html_content = element.get('html', '')
                
                # Извлекаем тексты моделей
                model_texts = extract_model_texts_from_html(html_content)
                
                # Нужно сопоставить животных с моделями
                # Это делается через choices в ranking вопросах
                
        # Ищем ranking вопрос чтобы получить маппинг animal -> model
        animal_to_model = {}
        for element in page.get('elements', []):
            if element.get('type') == 'ranking':
                for choice in element.get('choices', []):
                    model_name = choice.get('value', '')
                    animal_name = choice.get('text', '')
                    animal_to_model[animal_name] = model_name
                break  # Достаточно одного ranking вопроса
        
        # Извлекаем длины
        for element in page.get('elements', []):
            if element.get('type') == 'html':
                html_content = element.get('html', '')
                model_texts = extract_model_texts_from_html(html_content)
                
                if model_texts and animal_to_model:
                    task_lengths[page_name] = {}
                    for animal, text in model_texts.items():
                        model = animal_to_model.get(animal)
                        if model:
                            task_lengths[page_name][model] = len(text)
    
    return task_lengths


def calculate_verbosity_scores(model_lengths: dict[str, int]) -> dict[str, float]:
    """
    Расчёт verbosity scores для моделей на основе длины их ответов.
    
    Шкала (равномерная для 4 моделей):
    - Самый длинный = 1.0
    - 2-й = 0.667
    - 3-й = 0.333
    - Самый короткий = 0.0
    """
    if not model_lengths:
        return {}
    
    # Сортируем модели по длине (от короткого к длинному)
    sorted_models = sorted(model_lengths.items(), key=lambda x: x[1])
    n = len(sorted_models)
    
    if n == 1:
        return {sorted_models[0][0]: 0.5}
    
    # Равномерное распределение от 0 до 1
    scores = {}
    for i, (model, _) in enumerate(sorted_models):
        scores[model] = i / (n - 1)  # 0, 0.333, 0.667, 1.0
    
    return scores


def calculate_verbosity_bias(
    ranking_questions: list[tuple[str, list[str]]], 
    form_data: dict
) -> dict[str, float]:
    """
    Расчёт verbosity bias - насколько респондент предпочитает длинные ответы.
    
    Возвращает:
    - verbosity_first: средний verbosity score модели на 1-м месте
    - verbosity_weighted: взвешенный verbosity (учитывает все позиции)
    - verbosity_correlation: корреляция между длиной и рангом
    """
    task_lengths = extract_task_model_lengths(form_data)
    
    if not task_lengths:
        return {
            'verbosity_first': 0.5,
            'verbosity_weighted': 0.5,
            'verbosity_correlation': 0.0
        }
    
    first_place_verbosities = []
    weighted_verbosities = []
    all_lengths = []
    all_ranks = []
    
    for question_name, ranking in ranking_questions:
        # Извлекаем task_id из названия вопроса (например, "A1_fluency" -> "A1")
        task_id = question_name.split('_')[0]
        
        if task_id not in task_lengths:
            continue
        
        model_lengths = task_lengths[task_id]
        verbosity_scores = calculate_verbosity_scores(model_lengths)
        
        if not verbosity_scores:
            continue
        
        # Verbosity модели на 1-м месте
        first_model = ranking[0]
        if first_model in verbosity_scores:
            first_place_verbosities.append(verbosity_scores[first_model])
        
        # Взвешенный verbosity (веса: 4, 3, 2, 1 для позиций 1-4)
        weights = [4, 3, 2, 1]
        weighted_sum = 0
        weight_total = 0
        for i, model in enumerate(ranking):
            if model in verbosity_scores:
                weighted_sum += verbosity_scores[model] * weights[i]
                weight_total += weights[i]
        
        if weight_total > 0:
            weighted_verbosities.append(weighted_sum / weight_total)
        
        # Для корреляции собираем длины и ранги
        for i, model in enumerate(ranking):
            if model in model_lengths:
                all_lengths.append(model_lengths[model])
                all_ranks.append(i + 1)  # Ранг 1-4
    
    # Рассчитываем корреляцию между длиной и рангом
    # Отрицательная корреляция = предпочитает длинные (чем длиннее, тем меньше ранг = лучше)
    correlation = 0.0
    if len(all_lengths) >= 4:
        corr, _ = stats.spearmanr(all_lengths, all_ranks)
        if not np.isnan(corr):
            correlation = -corr  # Инвертируем: положительное = предпочитает длинные
    
    return {
        'verbosity_first': np.mean(first_place_verbosities) if first_place_verbosities else 0.5,
        'verbosity_weighted': np.mean(weighted_verbosities) if weighted_verbosities else 0.5,
        'verbosity_correlation': correlation
    }


def analyze_respondent(response_file: Path, form_file: Path) -> dict[str, Any]:
    """
    Анализ одного респондента.
    """
    response_data = load_json(response_file)
    form_data = load_json(form_file)
    
    # Извлекаем ответы
    responses = response_data.get('responses', {})
    
    # Получаем ranking вопросы
    ranking_questions = get_ranking_questions(responses)
    
    # Базовая информация
    result = {
        'respondent_id': response_file.stem,
        'task_id': response_data.get('taskId', ''),
        'timestamp': response_data.get('timestamp', ''),
        'num_questions': len(ranking_questions),
        'user_agent': response_data.get('userAgent', '')[:50] + '...' if len(response_data.get('userAgent', '')) > 50 else response_data.get('userAgent', ''),
    }
    
    # Sanity check результаты
    sanity_1 = responses.get('sanity_check_1', '')
    sanity_2 = responses.get('sanity_check_2', '')
    result['sanity_check_1_passed'] = sanity_1 == '3'  # Правильный ответ
    result['sanity_check_2_passed'] = sanity_2 == '2'  # Правильный ответ
    result['sanity_checks_passed'] = result['sanity_check_1_passed'] and result['sanity_check_2_passed']
    
    # Model bias - средний балл каждой модели
    model_bias = calculate_model_bias(ranking_questions)
    for model, avg_score in model_bias.items():
        short_name = model.replace('-', '_').replace('.', '_')
        result[f'model_bias_{short_name}'] = round(avg_score, 3)
    
    # Monotonicity score
    result['monotonicity_score'] = round(calculate_monotonicity_score(ranking_questions), 3)
    
    # Variance score
    result['variance_score'] = round(calculate_variance_score(ranking_questions), 3)
    
    # Verbosity bias
    verbosity_metrics = calculate_verbosity_bias(ranking_questions, form_data)
    result['verbosity_first'] = round(verbosity_metrics['verbosity_first'], 3)
    result['verbosity_weighted'] = round(verbosity_metrics['verbosity_weighted'], 3)
    result['verbosity_correlation'] = round(verbosity_metrics['verbosity_correlation'], 3)
    
    # Дополнительные метрики
    # Стандартное отклонение model bias (показывает разброс предпочтений)
    model_bias_values = list(model_bias.values())
    result['model_bias_std'] = round(float(np.std(model_bias_values)), 3)
    
    # Максимальная и минимальная модель
    if model_bias_values:
        max_model = max(model_bias.items(), key=lambda x: x[1])
        min_model = min(model_bias.items(), key=lambda x: x[1])
        result['preferred_model'] = max_model[0]
        result['least_preferred_model'] = min_model[0]
        result['preference_gap'] = round(max_model[1] - min_model[1], 3)
    
    return result


def interpret_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет интерпретацию результатов.
    """
    # Флаги для выявления проблемных респондентов
    
    # Высокий monotonicity score (>0.7) = механическое оценивание
    df['flag_mechanical_pattern'] = df['monotonicity_score'] > 0.7
    
    # Очень низкая дисперсия (<1.0) = нет различий в оценках
    # При идеальном ранжировании дисперсия = 1.25
    df['flag_low_variance'] = df['variance_score'] < 1.0
    
    # Сильное смещение к одной модели (preference_gap > 1.5)
    df['flag_strong_model_bias'] = df['preference_gap'] > 1.5
    
    # Провал sanity check = невнимательное заполнение
    df['flag_failed_sanity'] = ~df['sanity_checks_passed']
    
    # Verbosity bias: сильное предпочтение длинных (>0.7) или коротких (<0.3) ответов
    df['flag_verbosity_long'] = df['verbosity_weighted'] > 0.7
    df['flag_verbosity_short'] = df['verbosity_weighted'] < 0.3
    df['flag_verbosity_bias'] = df['flag_verbosity_long'] | df['flag_verbosity_short']
    
    # Общий флаг подозрительного респондента
    df['suspicious'] = (
        df['flag_mechanical_pattern'] | 
        df['flag_low_variance'] | 
        df['flag_failed_sanity']
    )
    
    return df


def main():
    """Основная функция анализа."""
    # Пути к данным
    base_path = Path(__file__).parent
    responses_path = base_path / 'responses'
    forms_path = base_path / 'initial_forms'
    output_path = base_path / 'bias_analysis_results.csv'
    
    print("=" * 60)
    print("Анализ смещения (bias) в ответах респондентов")
    print("=" * 60)
    
    # Собираем все файлы ответов
    response_files = sorted(responses_path.glob('*.json'))
    
    if not response_files:
        print("Не найдены файлы ответов!")
        return
    
    print(f"\nНайдено файлов ответов: {len(response_files)}")
    
    # Анализируем каждого респондента
    results = []
    for response_file in response_files:
        form_file = forms_path / response_file.name
        
        if not form_file.exists():
            print(f"Предупреждение: не найдена форма для {response_file.name}")
            continue
        
        try:
            result = analyze_respondent(response_file, form_file)
            results.append(result)
            print(f"  ✓ Обработан: {response_file.name}")
        except Exception as e:
            print(f"  ✗ Ошибка при обработке {response_file.name}: {e}")
    
    # Создаём DataFrame
    df = pd.DataFrame(results)
    
    # Добавляем интерпретацию
    df = interpret_results(df)
    
    # Сохраняем в CSV
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n✓ Результаты сохранены в: {output_path}")
    
    # Выводим статистику
    print("\n" + "=" * 60)
    print("СВОДНАЯ СТАТИСТИКА")
    print("=" * 60)
    
    print(f"\nВсего респондентов: {len(df)}")
    print(f"Прошли sanity check: {df['sanity_checks_passed'].sum()} ({df['sanity_checks_passed'].mean()*100:.1f}%)")
    
    print("\n--- Model Bias (средний балл модели по всем респондентам) ---")
    model_cols = [col for col in df.columns if col.startswith('model_bias_') and not col.endswith('_std')]
    for col in model_cols:
        model_name = col.replace('model_bias_', '').replace('_', '-')
        mean_score = df[col].mean()
        std_score = df[col].std()
        print(f"  {model_name}: {mean_score:.3f} ± {std_score:.3f}")
    
    print("\n--- Monotonicity Score ---")
    print(f"  Среднее: {df['monotonicity_score'].mean():.3f}")
    print(f"  Медиана: {df['monotonicity_score'].median():.3f}")
    print(f"  Макс: {df['monotonicity_score'].max():.3f}")
    print(f"  Респондентов с механическим паттерном (>0.7): {df['flag_mechanical_pattern'].sum()}")
    
    print("\n--- Variance Score ---")
    print(f"  Среднее: {df['variance_score'].mean():.3f}")
    print(f"  Медиана: {df['variance_score'].median():.3f}")
    print(f"  Мин: {df['variance_score'].min():.3f}")
    print(f"  Респондентов с низкой дисперсией (<1.0): {df['flag_low_variance'].sum()}")
    
    print("\n--- Verbosity Bias (предпочтение длины ответа) ---")
    print("  Шкала: 0.0 = короткие, 0.5 = нейтрально, 1.0 = длинные")
    print(f"  Среднее (1-е место): {df['verbosity_first'].mean():.3f}")
    print(f"  Среднее (взвешенное): {df['verbosity_weighted'].mean():.3f}")
    print(f"  Корреляция длина-ранг: {df['verbosity_correlation'].mean():.3f}")
    print(f"  Предпочитают длинные (>0.7): {df['flag_verbosity_long'].sum()}")
    print(f"  Предпочитают короткие (<0.3): {df['flag_verbosity_short'].sum()}")
    
    print("\n--- Подозрительные респонденты ---")
    suspicious = df[df['suspicious']]
    print(f"  Всего подозрительных: {len(suspicious)}")
    if len(suspicious) > 0:
        print("  Список:")
        for _, row in suspicious.iterrows():
            reasons = []
            if row['flag_mechanical_pattern']:
                reasons.append(f"механический паттерн ({row['monotonicity_score']:.2f})")
            if row['flag_low_variance']:
                reasons.append(f"низкая дисперсия ({row['variance_score']:.2f})")
            if row['flag_failed_sanity']:
                reasons.append("провал sanity check")
            print(f"    - {row['respondent_id']}: {', '.join(reasons)}")
    
    print("\n--- Предпочитаемые модели ---")
    preferred = df['preferred_model'].value_counts()
    for model, count in preferred.items():
        print(f"  {model}: {count} респондентов ({count/len(df)*100:.1f}%)")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
