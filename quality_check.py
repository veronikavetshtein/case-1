"""Пайплайн проверки качества модели: эталонный бенчмарк (Gold Benchmark),
формирование выборки для ручной разметки и расчёт Precision / Recall / F1 / точных цитат.

Использование из командной строки:
  py quality_check.py benchmark --provider auto
  py quality_check.py sample --csv demo_results.csv --size 30 --output gold_template_PRIVATE.csv
  py quality_check.py evaluate --file gold_template_PRIVATE.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

from criteria import ALL_CRITERIA
from llm_client import LocalLLM
from reviews import as_bool

CORE = list(ALL_CRITERIA)
TYPES = CORE + ['danger_detected', 'rules_detected', 'burnout_marker_detected']

# Встроенный эталонный набор (Gold Benchmark) для автоматической проверки новой модели или провайдера
GOLD_BENCHMARK = [
    {
        'id': 'BM01',
        'title': 'Личностное развитие и хобби (упоминание без динамики, score=1)',
        'text': 'Мы вместе посмотрели фильм в кинотеатре и обсудили сюжет за чаем в кафе.',
        'expected_criteria': {'hobbies': 1, 'household': 1},
        'expected_burnout': False,
        'expected_danger': False,
    },
    {
        'id': 'BM02',
        'title': 'Эмоциональная регуляция и явная динамика подопечного (score=2)',
        'text': (
            'Раньше когда у Марка что-то не получалось в настольной игре, он сильно психовал и бросал карточки, '
            'а сегодня когда он проиграл, он расстроился, но очень достойно себя вёл и спокойно продолжил общение.'
        ),
        'expected_criteria': {'emotions': 2, 'hobbies': 1, 'social': 1},
        'expected_burnout': False,
        'expected_danger': False,
    },
    {
        'id': 'BM03',
        'title': 'Ловушка субъекта: ребёнок не хочет встречаться (НЕ выгорание наставника)',
        'text': 'Подопечный сказал, что не хочет встречаться со мной и быстро устал. Я спокойно предложила обсудить причины.',
        'expected_criteria': {'social': 1},
        'expected_burnout': False,
        'expected_danger': False,
        'is_subject_trap': True,
    },
    {
        'id': 'BM04',
        'title': 'Маркер выгорания наставника №3 и №5 (растерянность и истощение автора отчёта)',
        'text': (
            'Вчера я не знала, как помочь подопечному. Мне кажется, я не справляюсь с ролью наставника. '
            'После встречи чувствую сильное опустошение и мне нужно время, чтобы восстановиться.'
        ),
        'expected_criteria': {},
        'expected_burnout': True,
        'expected_burnout_markers': [3, 5],
        'expected_danger': False,
    },
    {
        'id': 'BM05',
        'title': 'Прямая угроза жизни и здоровью (красный флаг опасности)',
        'text': 'Подопечный рассказал, что старшие ребята угрожали физической расправой за территорией центра.',
        'expected_criteria': {'safety': 1},
        'expected_burnout': False,
        'expected_danger': True,
    },
    {
        'id': 'BM06',
        'title': 'Бытовое обсуждение безопасности (НЕ красный флаг опасности)',
        'text': 'Во время поездки в метро строили маршруты через навигатор и обсуждали безопасность в общественных местах.',
        'expected_criteria': {'safety': 1, 'household': 1},
        'expected_burnout': False,
        'expected_danger': False,
        'is_subject_trap': True,
    },
    {
        'id': 'BM07',
        'title': 'Учёба, поступление в колледж и бытовые навыки (динамика score=2)',
        'text': (
            'Маша поступила в колледж и теперь студентка колледжа. В эту пятницу мы приготовили шарлотку '
            'и обсудили её планы на будущее и расписание занятий.'
        ),
        'expected_criteria': {'study': 2, 'household': 1, 'future': 1},
        'expected_burnout': False,
        'expected_danger': False,
    },
    {
        'id': 'BM08',
        'title': 'Трудоустройство и финансовая грамотность (динамика score=2)',
        'text': (
            'Володя устроился курьером, получил первую заработную плату и составил бюджет расходов на месяц, '
            'чтобы откладывать сбережения.'
        ),
        'expected_criteria': {'employment': 2, 'finance': 2},
        'expected_burnout': False,
        'expected_danger': False,
    },
    {
        'id': 'BM09',
        'title': 'Здоровье, риск ПНИ и неоказание медпомощи (красный флаг опасности + правовая грамотность)',
        'text': (
            'Зрение -8 и разрыв связок в коленном суставе не оперируют. Нельзя допустить, чтобы её определили после 18 в ПНИ.'
        ),
        'expected_criteria': {'health': 1, 'legal': 1},
        'expected_burnout': False,
        'expected_danger': True,
    },
    {
        'id': 'BM10',
        'title': 'Отношения с родственниками, курение и работа с зависимостями',
        'text': (
            'Поговорили с подопечным про его ссору с мамой и бабушкой, а также обсудили вред курения вейпа '
            'и способы бросить курить.'
        ),
        'expected_criteria': {'family_rel': 1, 'addictions': 1},
        'expected_burnout': False,
        'expected_danger': False,
    },
]


def run_benchmark(llm: LocalLLM | None = None, provider: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Прогоняет эталонный набор GOLD_BENCHMARK через выбранную модель/провайдера и считает метрики."""
    client = llm or LocalLLM(provider_override=provider)
    rows = []
    tp_crit = fp_crit = fn_crit = 0
    exact_score_hits = total_score_checks = 0
    valid_quotes = total_positive_quotes = 0
    burnout_correct = danger_correct = trap_correct = trap_total = 0

    for item in GOLD_BENCHMARK:
        text = item['text']
        analysis = client.analyze_report(text)
        flat_crit = {
            k: v for grp in analysis['criteria'].values() for k, v in grp.items() if isinstance(v, dict)
        }
        flags = analysis['red_flags']
        exp_crit = item['expected_criteria']

        # Проверяем критерии
        pred_pos = []
        for k in CORE:
            p_score = flat_crit.get(k, {}).get('score') or 0
            p_quote = str(flat_crit.get(k, {}).get('quote') or '').strip()
            g_score = exp_crit.get(k, 0)
            p_bin = 1 if p_score >= 1 else 0
            g_bin = 1 if g_score >= 1 else 0
            if p_bin == 1 and g_bin == 1:
                tp_crit += 1
            elif p_bin == 1 and g_bin == 0:
                fp_crit += 1
            elif p_bin == 0 and g_bin == 1:
                fn_crit += 1
            if g_score > 0:
                total_score_checks += 1
                if p_score == g_score:
                    exact_score_hits += 1
            if p_score > 0:
                pred_pos.append(f'{k}={p_score}')
                total_positive_quotes += 1
                if p_quote and p_quote in text:
                    valid_quotes += 1

        p_burn = bool(flags.get('burnout', {}).get('detected', False))
        p_dang = bool(flags.get('danger', {}).get('detected', False))
        b_ok = p_burn == item['expected_burnout']
        d_ok = p_dang == item['expected_danger']
        if b_ok:
            burnout_correct += 1
        if d_ok:
            danger_correct += 1
        if item.get('is_subject_trap'):
            trap_total += 1
            if b_ok and d_ok:
                trap_correct += 1

        rows.append({
            'ID': item['id'],
            'Тестовый сценарий': item['title'],
            'Ожидаемые критерии': ', '.join(f'{k}={v}' for k, v in exp_crit.items()) or '—',
            'Найдено моделью': ', '.join(pred_pos) or '—',
            'Выгорание (ожид./факт)': f'{item["expected_burnout"]} / {p_burn}',
            'Опасность (ожид./факт)': f'{item["expected_danger"]} / {p_dang}',
            'Статус флагов': '✅ OK' if (b_ok and d_ok) else '❌ Ошибка',
        })

    precision = tp_crit / (tp_crit + fp_crit) if (tp_crit + fp_crit) else 1.0
    recall = tp_crit / (tp_crit + fn_crit) if (tp_crit + fn_crit) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    summary = {
        'backend': client._resolve_backend(),
        'model': client.model if client._resolve_backend() != 'offline' else 'OfflineRussianAnalyzer (локальный NLP)',
        'cases_count': len(GOLD_BENCHMARK),
        'criteria_precision': round(precision, 3),
        'criteria_recall': round(recall, 3),
        'criteria_f1': round(f1, 3),
        'score_exact_accuracy': round(exact_score_hits / max(1, total_score_checks), 3),
        'quote_faithfulness': round(valid_quotes / max(1, total_positive_quotes), 3),
        'burnout_accuracy': round(burnout_correct / len(GOLD_BENCHMARK), 3),
        'danger_accuracy': round(danger_correct / len(GOLD_BENCHMARK), 3),
        'subject_trap_pass_rate': round(trap_correct / max(1, trap_total), 3),
    }
    return pd.DataFrame(rows), summary


def create_sample(src, out, n=30):
    df = pd.read_csv(src, dtype={'volunteer_id': str, 'case_id': str}, keep_default_na=False)
    df = df[df.analysis_status == 'ok'].copy()
    if df.empty:
        raise ValueError('Нет успешно обработанных отчётов для оценки')
    sample = df.sample(min(n, len(df)), random_state=42)
    rows = []
    for _, r in sample.iterrows():
        item = {k: r.get(k, '') for k in ('source_row', 'case_id', 'volunteer_id', 'date', 'text')}
        for k in CORE:
            item['pred_' + k] = int(pd.to_numeric(r.get(k + '_score', 0), errors='coerce') >= 1)
            item['gold_' + k] = ''
        for k in TYPES[len(CORE):]:
            item['pred_' + k] = int(as_bool(r.get(k, False)))
            item['gold_' + k] = ''
        rows.append(item)
    pd.DataFrame(rows).to_csv(out, index=False, encoding='utf-8-sig')
    print('Разметьте gold_... вручную значениями 0 или 1, не угадывая по pred_...:', out)


def evaluate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    metrics = []
    for k in TYPES:
        p = 'pred_' + k
        g = 'gold_' + k
        if p not in df or g not in df:
            continue
        both = df[df[g].astype(str).isin(['0', '1'])]
        if both.empty:
            continue
        pred = both[p].astype(int)
        gold = both[g].astype(int)
        tp = int(((pred == 1) & (gold == 1)).sum())
        fp = int(((pred == 1) & (gold == 0)).sum())
        fn = int(((pred == 0) & (gold == 1)).sum())
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = (2 * precision * recall / (precision + recall)) if (precision and recall) else None
        metrics.append({
            'criterion': k,
            'n_labeled': len(both),
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'precision': round(precision, 3) if precision is not None else None,
            'recall': round(recall, 3) if recall is not None else None,
            'f1': round(f1, 3) if f1 is not None else None,
        })
    return pd.DataFrame(metrics)


def evaluate(path):
    df = pd.read_csv(path, keep_default_na=False)
    res = evaluate_dataframe(df)
    if res.empty:
        print('Разметка пуста. Заполните столбцы gold_... значениями 0/1.')
        return
    print(res.to_string(index=False))


def main():
    p = argparse.ArgumentParser(description='Пайплайн проверки качества модели')
    sub = p.add_subparsers(dest='action', required=True)
    s = sub.add_parser('sample')
    s.add_argument('--csv', default='demo_results.csv')
    s.add_argument('--size', type=int, default=30)
    s.add_argument('--output', default='gold_template_PRIVATE.csv')

    e = sub.add_parser('evaluate')
    e.add_argument('--file', default='gold_template_PRIVATE.csv')

    b = sub.add_parser('benchmark')
    b.add_argument('--provider', default=None, help='auto | ollama | openai | offline')

    a = p.parse_args()
    if a.action == 'sample':
        create_sample(a.csv, a.output, a.size)
    elif a.action == 'evaluate':
        evaluate(a.file)
    elif a.action == 'benchmark':
        df_res, summary = run_benchmark(provider=a.provider)
        print('=== РЕЗУЛЬТАТЫ ЭТАЛОННОГО БЕНЧМАРКА ===')
        for k, v in summary.items():
            print(f'  {k}: {v}')
        print('\n' + df_res.to_string(index=False))


if __name__ == '__main__':
    main()
