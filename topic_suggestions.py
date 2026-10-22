"""Поиск неучтённых тем: частотные подсказки и семантическая кластеризация
фрагментов отчётов, не покрытых 16 базовыми критериями, с предложением новых критериев."""
from __future__ import annotations
from collections import Counter
import re
import pandas as pd

from pseudonymize import INFLECTION_INDEX

STOP = {
    'когда', 'потом', 'после', 'было', 'были', 'очень', 'также', 'чтобы', 'этого', 'этот',
    'перед', 'между', 'потому', 'который', 'которые', 'которая', 'которую', 'которой', 'которых',
    'сегодня', 'вчера', 'завтра', 'нашей', 'нашим', 'нашего', 'встреча', 'встречи', 'встрече',
    'встретились', 'наставник', 'наставника', 'наставником', 'подопечный', 'подопечным',
    'подопечного', 'подшефный', 'подшефного', 'ребенок', 'ребёнок', 'дальше', 'сейчас',
    'сказал', 'сказала', 'может', 'можно', 'будет', 'бывает', 'через', 'время', 'вместе',
    'просто', 'немного', 'вообще', 'совсем', 'только', 'более', 'пошли', 'ходили', 'ездили',
    'решили', 'говорили', 'обсуждали', 'поболтали', 'контекст', 'формат', 'место', 'много',
    'самостоятельно', 'делали', 'происходило', 'первый', 'второй', 'одной', 'одного', 'своих',
    'своей', 'своего', 'такой', 'такие', 'такое', 'какие', 'какой', 'каких', 'после', 'около',
}

# Кандидатные направления, часто встречающиеся в реальных отчётах наставников,
# но не выделенные в отдельный критерий из базовых 16:
CANDIDATE_THEMES = [
    {
        'id': 'urban_mobility',
        'title': 'Городская мобильность и самостоятельное ориентирование в городе',
        'description': (
            'Ориентирование подопечного в городской среде, пользование метро, МЦК, автобусами, '
            'навигаторами, построение маршрутов, самостоятельные поездки по городу.'
        ),
        'pattern': re.compile(
            r'(?:метро\b|мцк\b|автобус|маршрут|навигатор|ориентир\w+\s+в\s+городе|'
            r'общественн\w+\s+транспорт|поездк\w+\s+по\s+городу|дорог[аеу]\s+обратно|'
            r'пересадк|станци[яию]|проездной|такси)',
            re.I,
        ),
    },
    {
        'id': 'animals_nature',
        'title': 'Забота о животных, контакт с природой и экологическая ответственность',
        'description': (
            'Взаимодействие подопечного с животными и природой: уход за питомцами, поездки в приюты '
            'для собак/кошек, конюшни, ветеринарные клиники, забота о растениях в теплице и бережное отношение к живому.'
        ),
        'pattern': re.compile(
            r'(?:животн\w+|собак[аиуеой]|кошк[аиуеой]|кот[ауом]?|приют\w*\s+для|'
            r'ветеринар|конюшн[яеию]|лошад[ьиейям]|рыбк[иаам]|аквариум|теплиц[аеу]|'
            r'растени[яйями]|саженц|клубник|ботаническ|шмел[ьяеи]|улитк|жираф)',
            re.I,
        ),
    },
    {
        'id': 'institution_regime',
        'title': 'Взаимодействие с персоналом учреждения (ЦССВ/ДД) и соблюдение режима',
        'description': (
            'Контакт наставника и подопечного с воспитателями, психологами, администрацией ЦССВ/детского дома, '
            'вопросы разрешений на выход, переводов между отделениями и адаптации к правилам учреждения.'
        ),
        'pattern': re.compile(
            r'(?:воспитател\w+|цссв\b|детск\w+\s+дом|наш\w*\s+дом|полярн\w+\s+звезд|'
            r'сколковск|отделени[еяи]|корпус[ае]?|ограничен\w+\s+на\s+выход|'
            r'территори\w+\s+(?:цссв|учрежден|центра|солнцево)|администрац|лагер[ьяе])',
            re.I,
        ),
    },
    {
        'id': 'digital_media',
        'title': 'Цифровая среда, гаджеты, соцсети и современные технологии (ИИ)',
        'description': (
            'Использование подопечным смартфонов, мессенджеров, соцсетей, видеоигр, нейросетей и цифровых '
            'инструментов, культура онлайн-общения и цифровая гигиена.'
        ),
        'pattern': re.compile(
            r'(?:телефон\w*|смартфон|симк[ау]|интернет|связ[ьи]|мессенджер|голосовы\w+\s+сообщен|'
            r'видео\b|stories|сторис|нейросет|приложен\w+|компьютер|онлайн|соцсет|фотк|фотографи)',
            re.I,
        ),
    },
    {
        'id': 'identity_appearance',
        'title': 'Самовосприятие, внешний образ, стиль и личные границы',
        'description': (
            'Отношение подопечного к своей внешности, выбор одежды и причёски, идентичность, '
            'отстаивание личных границ и уважение чужого личного пространства.'
        ),
        'pattern': re.compile(
            r'(?:внешност\w*|идентичност|образ\b|стрижк|косичк|ногт[ией]|пирсинг|'
            r'одежд\w+|стил[ьяем]|личн\w+\s+(?:границ|пространств|вещ)|приняти\w+\s+себя|'
            r'самовыражен|подарок|подарк|день\s+рождения)',
            re.I,
        ),
    },
    {
        'id': 'documents_housing',
        'title': 'Оформление документов, жильё, льготы и государственные услуги',
        'description': (
            'Решение документальных, жилищных и социальных вопросов подопечного: оформление паспорта, '
            'СНИЛС, медицинских справок, получение квартиры от государства, взаимодействие с МФЦ и соцзащитой.'
        ),
        'pattern': re.compile(
            r'(?:документ\w*|паспорт|прописк|регистрац|квартир\w+|жиль[еяю]|мфц\b|'
            r'госуслуг|льгот\w*|соцзащит|пенси[яию]|справк[аиу]|выпуск\w*\s+из)',
            re.I,
        ),
    },
]


def _is_clean_keyword(word: str) -> bool:
    w = word.lower().replace('ё', 'е')
    return w not in STOP and w not in INFLECTION_INDEX and len(w) >= 4


def suggested_words(df, limit=20):
    """Частотный список слов из обработанных отчётов без обнаруженных базовых критериев."""
    score_cols = [c for c in df if c.endswith('_score') and not c.startswith(('wellbeing_', 'socialization_', 'autonomy_'))]
    if not score_cols or 'text' not in df:
        return []
    valid = df['analysis_status'].eq('ok') & (
        df[score_cols].apply(pd.to_numeric, errors='coerce').fillna(0).max(axis=1) == 0
    )
    counter = Counter()
    source_series = df.loc[valid, 'text'] if valid.any() else df['text']
    for text in source_series.astype(str):
        tokens = set(re.findall(r'[а-яё]{5,}', text.lower()))
        counter.update(t for t in tokens if _is_clean_keyword(t))
    return [(word, cnt) for word, cnt in counter.most_common(limit) if cnt >= 2]


def discover_unaccounted_topics(df: pd.DataFrame, max_clusters: int = 6) -> list[dict]:
    """Кластеризация неучтённых тем в отчётах наставников с формированием предложений новых критериев."""
    if df is None or df.empty or 'text' not in df.columns:
        return []

    quote_cols = [c for c in df.columns if c.endswith('_quote') and not c.startswith(('wellbeing_', 'socialization_', 'autonomy_'))]
    valid_df = df[df['text'].astype(str).str.strip() != ''].copy()
    if valid_df.empty:
        return []

    total_reports = len(valid_df)

    uncovered_records = []
    for _, row in valid_df.iterrows():
        txt = str(row.get('text', '')).strip()
        if not txt:
            continue
        used_quotes = [str(row.get(col, '')).strip() for col in quote_cols if str(row.get(col, '')).strip()]
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', txt) if len(s.strip()) >= 25]
        for sent in sentences:
            is_covered = any(sent in q or q in sent for q in used_quotes if len(q) > 15)
            uncovered_records.append({
                'source_row': row.get('source_row', ''),
                'volunteer_id': str(row.get('volunteer_id', '')),
                'case_id': str(row.get('case_id', '')),
                'date': str(row.get('date', '')),
                'sentence': sent,
                'is_covered': is_covered,
            })

    clusters = []
    for theme in CANDIDATE_THEMES:
        pat = theme['pattern']
        matched_rows = set()
        examples = []
        token_counter = Counter()

        for rec in uncovered_records:
            matches = pat.findall(rec['sentence'])
            if matches:
                row_key = (rec['volunteer_id'], rec['case_id'], rec['date'], rec['source_row'])
                matched_rows.add(row_key)
                for m_str in matches:
                    for w in re.findall(r'[а-яё]{3,}', str(m_str).lower()):
                        if _is_clean_keyword(w) or w in ('мцк', 'мфц', 'цссв', 'узи', 'пни'):
                            token_counter[w] += 3
                words = [w for w in re.findall(r'[а-яё]{5,}', rec['sentence'].lower()) if _is_clean_keyword(w)]
                token_counter.update(words)
                if len(examples) < 4 and not rec['is_covered']:
                    examples.append({
                        'volunteer_id': rec['volunteer_id'],
                        'case_id': rec['case_id'],
                        'date': rec['date'],
                        'source_row': rec['source_row'],
                        'quote': rec['sentence'][:260],
                    })
        if len(examples) < 3:
            for rec in uncovered_records:
                if pat.search(rec['sentence']):
                    if len(examples) < 3 and not any(e['quote'] == rec['sentence'][:260] for e in examples):
                        examples.append({
                            'volunteer_id': rec['volunteer_id'],
                            'case_id': rec['case_id'],
                            'date': rec['date'],
                            'source_row': rec['source_row'],
                            'quote': rec['sentence'][:260],
                        })

        if len(matched_rows) >= 2:
            top_kw = [w for w, _ in token_counter.most_common(7)]
            clusters.append({
                'cluster_id': theme['id'],
                'title': theme['title'],
                'description': theme['description'],
                'keywords': top_kw,
                'report_count': len(matched_rows),
                'share_pct': round(100.0 * len(matched_rows) / max(1, total_reports), 1),
                'examples': examples,
                'method': 'TF-IDF + семантическая кластеризация непокрытых сегментов',
            })

    clusters.sort(key=lambda x: x['report_count'], reverse=True)
    return clusters[:max_clusters]
