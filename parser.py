"""Импорт исходной 1С-выгрузки без потери пустых строк отчётов."""
from pathlib import Path
import re
import pandas as pd

ACTIVITY = 'Работа с подопечными в учреждениях'


def _value(value):
    if pd.isna(value):
        return ''
    return str(value).strip()


def _code(value):
    if pd.isna(value):
        return ''
    if isinstance(value, (float, int)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def parse_excel(file_path):
    """Возвращает ВСЕ активности (включая пустые) для проверки последовательности."""
    path = Path(file_path)
    raw = pd.read_excel(path, header=None, dtype=object)
    if raw.shape[1] < 7:
        raise ValueError('Неожиданный формат Excel: ожидалось минимум 7 столбцов.')
    data = []
    case_id = ''
    for idx, row in raw.iterrows():
        first = _value(row.iloc[0]).replace('\xa0', ' ')
        if re.match(r'^Кейс\s*№', first, flags=re.I):
            case_id = ' '.join(first.split())
            continue
        if first != ACTIVITY:
            continue
        # В реальной выгрузке 1С: дата в D (index 3), комментарий в F (index 5), код волонтёра в G.
        report_text = _value(row.iloc[5])
        raw_date = row.iloc[3]
        date = pd.to_datetime(raw_date, errors='coerce', dayfirst=True)
        data.append({
            'source_row': idx + 1,  # номер строки Excel (нумерация с 1)
            'case_id': case_id,
            'volunteer_id': _code(row.iloc[6]),
            'date': date.strftime('%Y-%m-%d') if pd.notna(date) else '',
            'date_raw': _value(raw_date),
            'text': report_text,
        })
    if not data:
        raise ValueError('В файле не найдены строки активности «Работа с подопечными в учреждениях».')
    df = pd.DataFrame(data)
    # Технические повторы — кандидаты на дубли; сохраняем строки, но не анализируем повторно.
    df['possible_duplicate'] = df.duplicated(
        subset=['case_id', 'volunteer_id', 'date', 'text'], keep='first'
    )
    return df


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('excel', nargs='?', default='reports.xlsx')
    args = p.parse_args()
    output = parse_excel(args.excel)
    output.to_pickle(Path(__file__).parent / 'parsed_data.pkl')
    print(f'Активностей: {len(output)}; пустых: {output.text.eq("").sum()}; возможных дублей: {output.possible_duplicate.sum()}')
