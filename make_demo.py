"""Только вымышленные данные! Создаёт demo_results.csv; перезаписывает файл."""
from pathlib import Path
import pandas as pd
from batch_process import make_blank, validate_schema
from burnout_rules import add_burnout_sequences

samples = [
    ('Кейс №ДЕМО1', 'DEMO01', '2026-10-01', 'Вчера я не знала, как помочь подопечному. Мне кажется, я не справляюсь с ролью наставника.', 3),
    ('Кейс №ДЕМО1', 'DEMO01', '2026-10-08', 'После встречи чувствую сильное опустошение и мне нужно время, чтобы восстановиться.', 5),
    ('Кейс №ДЕМО1', 'DEMO01', '2026-10-15', 'Мы вместе посмотрели фильм и обсудили его.', None),
    ('Кейс №ДЕМО2', 'DEMO02', '2026-10-02', 'Подопечный сказал, что не хочет встречаться со мной. Я предложила обсудить причины.', None),
    ('Кейс №ДЕМО2', 'DEMO02', '2026-10-09', 'Подопечный рассказал, что ему угрожали физической расправой.', None),
]
rows = []
for n, (case, vol, date, text, marker) in enumerate(samples, 1):
    obj = make_blank({'source_row': n, 'case_id': case, 'volunteer_id': vol,
                      'date': date, 'date_raw': date, 'text': text})
    obj['is_demo'] = True
    obj['analysis_status'] = 'ok'
    obj['red_flags_status'] = 'ok'
    if marker:
        obj['burnout_marker_detected'] = True
        obj['burnout_markers'] = f'[{marker}]'
        obj['burnout_quote'] = text
    if n == 5:
        obj['danger_detected'] = True
        obj['danger_quote'] = 'ему угрожали физической расправой'
        obj['danger_reason'] = 'Упомянута угроза насилия; нужна проверка координатора.'
    if n == 3:
        obj['hobbies_score'] = 1
        obj['hobbies_quote'] = 'посмотрели фильм и обсудили его'
    rows.append(obj)

result = validate_schema(add_burnout_sequences(pd.DataFrame(rows)))
path = Path(__file__).parent / 'demo_results.csv'
result.to_csv(path, index=False, encoding='utf-8-sig')
print('Демонстрационные результаты созданы:', path)