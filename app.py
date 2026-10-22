from __future__ import annotations
import io
import json
import tempfile
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

from batch_process import run_batch_analysis
from criteria import ALL_CRITERIA, BURNOUT_MARKERS, CRITERIA_GROUPS, SUMMARY_QUESTIONS
from criteria_store import load_config, save_config, active_criteria, criteria_groups
from parser import parse_excel
from pseudonymize import (
    pseudonymize_detailed,
    replace_explicit,
    restore,
    save_private_map,
    load_private_map,
    DEFAULT_VAULT,
)
from quality_check import run_benchmark, create_sample, evaluate_dataframe
from reviews import (
    signals,
    signal_export,
    crm_volunteer_summary_export,
    list_reviews,
    set_review,
    STATUSES,
    as_bool,
)
from topic_suggestions import suggested_words, discover_unaccounted_topics

HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'demo_results.csv'
EXCEL = HERE / 'reports.xlsx'
POLICY_FILE = HERE / 'rules_policy.txt'

st.set_page_config(page_title='Наставник.Аналитика', page_icon='📊', layout='wide')
st.title('Наставник.Аналитика')
st.caption(
    'Аналитическая система для координаторов фонда «Волонтёры в помощь детям-сиротам» · '
    '16 критериев + 3 обобщающих вопроса (17–19) · 3 типа красных флагов · обратимая псевдонимизация · экспорт в 1С CRM'
)


def safe_csv(frame: pd.DataFrame) -> bytes:
    """Prefix risky Excel formula strings before generating a CSV."""
    copy = frame.copy()
    for col in copy.select_dtypes(include='object').columns:
        copy[col] = copy[col].map(
            lambda s: "'" + s if isinstance(s, str) and s.lstrip().startswith(('=', '+', '-', '@')) else s
        )
    return copy.to_csv(index=False).encode('utf-8-sig')


def safe_excel(frame: pd.DataFrame, sheet_name: str = 'CRM_1C') -> bytes:
    """Формирует .xlsx файл в памяти для прямой загрузки в 1С / Excel."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        frame.to_excel(writer, index=False, sheet_name=sheet_name)
    return buf.getvalue()


@st.cache_data(show_spinner=False)
def read_csv_cached(path: str, timestamp: int) -> pd.DataFrame:
    return pd.read_csv(path, dtype={'volunteer_id': str, 'case_id': str}, keep_default_na=False)


def numeric(df: pd.DataFrame, key: str) -> pd.Series:
    col = key + '_score'
    return pd.to_numeric(df[col], errors='coerce') if col in df else pd.Series(float('nan'), index=df.index)


def bools(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col].map(as_bool) if col in df else pd.Series(False, index=df.index)


def maybe_mask_text(text: str, mask_enabled: bool) -> str:
    if not mask_enabled or not str(text).strip():
        return str(text)
    masked, _, _ = pseudonymize_detailed(str(text))
    return masked


def render_criteria(group, df: pd.DataFrame, only_dynamics: bool = False, mask_names: bool = False):
    config = {i['key']: i for i in load_config()}
    for key in group:
        label = config[key]['title'] if key in config else key
        score = numeric(df, key)
        n = int((score >= 1).sum())
        changed = int((score == 2).sum())
        missing = int(score.isna().sum())
        with st.expander(f'{label} — упоминания: {n}; описанная динамика (score=2): {changed}', expanded=False):
            st.caption(config.get(key, {}).get('description', ''))
            if missing:
                st.warning(f'Для {missing} записей оценка отсутствует или отчёт пуст.')
            subset = df.loc[score == 2] if only_dynamics else df.loc[score >= 1]
            if subset.empty:
                st.info('Нет найденных подтверждаемых цитат в выбранной выборке.')
            else:
                for _, r in subset.head(80).iterrows():
                    evidence = str(r.get(key + '_quote', '')).strip()
                    if not evidence or evidence not in str(r.get('text', '')):
                        st.warning(f'Цитата не совпадает с текстом; исходная строка {r.get("source_row", "?")}')
                        continue
                    is_dyn = pd.to_numeric(r.get(key + '_score'), errors='coerce') == 2
                    badge = '🟢 Явно описано изменение (score=2)' if is_dyn else '🔵 Упоминание / действие (score=1)'
                    st.markdown(
                        f'**Волонтёр {r.get("volunteer_id")} · {r.get("case_id")} · {r.get("date")} · '
                        f'строка Excel {r.get("source_row", "?")} · {badge}**'
                    )
                    disp_quote = maybe_mask_text(evidence, mask_names)
                    st.write('«' + disp_quote + '»')
                if len(subset) > 80:
                    st.caption(f'Показано 80 из {len(subset)} цитат. Используйте фильтры и экспорт.')


# --- БОКОВАЯ ПАНЕЛЬ: ДАННЫЕ И ЗАПУСК АНАЛИЗА ---
st.sidebar.header('Исходные данные')
upload_csv = st.sidebar.file_uploader('Загрузить CSV результатов (необязательно)', type=['csv'], key='upload_results')
if upload_csv is not None:
    try:
        df = pd.read_csv(upload_csv, dtype={'volunteer_id': str, 'case_id': str}, keep_default_na=False)
    except Exception as exc:
        st.error(f'Ошибка CSV: {exc}')
        st.stop()
elif RESULTS.exists():
    df = read_csv_cached(str(RESULTS), RESULTS.stat().st_mtime_ns)
else:
    df = pd.DataFrame()

with st.sidebar.expander('🚀 Запуск анализа / Новая выгрузка Excel', expanded=False):
    st.caption('Загрузите новую выгрузку 1С (.xlsx) или перезапустите анализ текущей базы прямо из браузера.')
    workbook = st.file_uploader('Файл выгрузки фонда (.xlsx)', type=['xlsx'], key='upload_xlsx')
    if workbook is not None:
        if st.button('Проверить и сохранить как reports.xlsx'):
            try:
                with tempfile.NamedTemporaryFile(suffix='.xlsx') as temporary:
                    temporary.write(workbook.getvalue())
                    temporary.flush()
                    parsed = parse_excel(temporary.name)
                EXCEL.write_bytes(workbook.getvalue())
                st.success(f'Excel сохранён: {len(parsed)} активностей ({parsed.case_id.nunique()} кейсов).')
            except Exception as exc:
                st.error(f'Не удалось прочитать выгрузку: {exc}')

    provider_label = st.selectbox(
        'Движок анализа',
        [
            ('auto', 'Авто (Ollama / API / Локальный NLP)'),
            ('offline', 'Локальный лингвистический анализатор (быстро, без GPU)'),
            ('ollama', 'Ollama (qwen2.5:3b-instruct)'),
            ('openai', 'OpenAI-совместимый API'),
        ],
        format_func=lambda x: x[1],
    )[0]
    limit_cases = st.number_input('Ограничить число кейсов (0 = все кейсы)', min_value=0, max_value=500, value=0, step=1)
    use_resume = st.checkbox('Использовать кэш неизменившихся отчётов (--resume)', value=True)
    if st.button('▶ Запустить анализ отчётов', type='primary', disabled=not EXCEL.exists()):
        prog = st.progress(0.0, text='Подготовка к анализу...')
        try:
            def _on_progress(done, total, row_data, status_str):
                prog.progress(done / max(1, total), text=f'Обработано {done}/{total}: {row_data.get("case_id", "")} ({status_str})')

            run_batch_analysis(
                excel_path=EXCEL,
                output_path=RESULTS,
                max_cases=int(limit_cases) if limit_cases > 0 else None,
                resume=use_resume,
                fast=True,
                provider=provider_label,
                progress_callback=_on_progress,
            )
            read_csv_cached.clear()
            st.success('Анализ успешно завершён! Данные обновлены.')
            st.rerun()
        except Exception as exc:
            st.error(f'Ошибка при анализе: {exc}')

if df.empty:
    st.info('Нет файла результатов. Нажмите «▶ Запустить анализ отчётов» в боковой панели или выполните `py batch_process.py --fast`.')
    st.stop()

mandatory = ['source_row', 'volunteer_id', 'case_id', 'date', 'text', 'analysis_status']
missing = [c for c in mandatory if c not in df.columns]
if missing:
    st.error('Некорректный или старый CSV. Не хватает столбцов: ' + ', '.join(missing))
    st.stop()

if 'is_demo' in df and bools(df, 'is_demo').any():
    st.info('СЕЙЧАС ОТКРЫТЫ ВЫМЫШЛЕННЫЕ ДЕМОНСТРАЦИОННЫЕ ДАННЫЕ')

status_counts = df['analysis_status'].astype(str).value_counts().to_dict()
not_ready = sum(status_counts.get(x, 0) for x in ('partial', 'error', 'unprocessed'))
if not_ready:
    st.warning(f'В текущем CSV {not_ready} отчётов не обработаны полностью. «Не обнаружено» нельзя интерпретировать как отсутствие проблемы.')

cfg = load_config()
all_active = active_criteria()
missing_scores = [key for key in all_active if key + '_score' not in df]
if missing_scores:
    st.warning('По этим новым критериям в текущем CSV ещё нет анализа (нажмите «▶ Запустить анализ отчётов»): ' + ', '.join(missing_scores))

# --- БОКОВАЯ ПАНЕЛЬ: ФИЛЬТРЫ ---
st.sidebar.divider()
st.sidebar.header('Фильтры и приватность')
mask_names_ui = st.sidebar.checkbox('🔒 Скрывать имена в интерфейсе (авто-псевдонимизация)', value=False)
vols = sorted(df['volunteer_id'].astype(str).unique().tolist())
vol = st.sidebar.selectbox('Наставник / код волонтёра', ['Все'] + vols)
filtered = df if vol == 'Все' else df[df.volunteer_id.astype(str) == vol]
cases = sorted(filtered.case_id.astype(str).unique().tolist())
case = st.sidebar.selectbox('Кейс', ['Все'] + cases)
filtered = filtered if case == 'Все' else filtered[filtered.case_id.astype(str) == case]
if 'date' in filtered:
    valid_dates = pd.to_datetime(filtered.date, errors='coerce').dropna()
    if len(valid_dates):
        beginning, end = valid_dates.min().date(), valid_dates.max().date()
        if beginning < end:
            date_range = st.sidebar.date_input('Период', value=(beginning, end), min_value=beginning, max_value=end)
            if isinstance(date_range, tuple) and len(date_range) == 2:
                dates = pd.to_datetime(filtered.date, errors='coerce').dt.date
                filtered = filtered[dates.between(date_range[0], date_range[1])]

search_query = st.sidebar.text_input('Поиск по тексту отчёта', value='').strip()
if search_query:
    filtered = filtered[filtered['text'].astype(str).str.contains(search_query, case=False, na=False)]

filtered = filtered.copy()

all_signals = signals(filtered)
reviews = list_reviews()
n_burnout = sum(item['kind'] == 'burnout' for item in all_signals)
n_danger = sum(item['kind'] == 'danger' for item in all_signals)
n_rules = sum(item['kind'] == 'rules' for item in all_signals)

score_cols_present = [f'{k}_score' for k in all_active if f'{k}_score' in filtered.columns]
if score_cols_present:
    any_dyn_report = (filtered[score_cols_present].apply(pd.to_numeric, errors='coerce').fillna(0) == 2).any(axis=1)
    n_dyn_reports = int(any_dyn_report.sum())
else:
    n_dyn_reports = 0

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric('Отчётов в выборке', len(filtered))
m2.metric('Обработано (непустых)', int(filtered.analysis_status.eq('ok').sum()))
m3.metric('Пар (волонтёр + кейс)', int(filtered[['volunteer_id', 'case_id']].drop_duplicates().shape[0]))
m4.metric('Отчётов с динамикой (score=2)', n_dyn_reports)
m5.metric('Серий выгорания (2+ подряд)', n_burnout)
m6.metric('Сигналов опасности', n_danger)

(
    t_dashboard,
    t_flags,
    t_one,
    t_two,
    t_three,
    t_extra,
    t_history,
    t_reviews,
    t_settings,
    t_quality,
    t_export,
) = st.tabs([
    'Обзор и графики',
    'Красные флаги',
    '1–5: Личностное',
    '6–9, 16: Практическое',
    '10–15: Здоровье и право',
    'Другие темы (Кластеры)',
    'История и первоисточник',
    'Проверка координатора',
    'Настройка критериев',
    'Качество и Псевдонимизация',
    'Экспорт для CRM (1С)',
])

with t_dashboard:
    st.subheader('Макро- и микро-аналитика по программе наставничества')
    st.caption('На графиках отображены подтверждённые цитатами упоминания (score=1) и явно описанные изменения в поведении/навыках подопечного (score=2).')
    a, b = st.columns([3, 2])
    with a:
        data = []
        for key in ALL_CRITERIA:
            scores = numeric(filtered, key)
            meta = next((item for item in cfg if item['key'] == key), {'title': key})
            data.append({
                'Критерий': meta['title'],
                'Упоминание / действие (1)': int((scores == 1).sum()),
                'Описанное изменение / динамика (2)': int((scores == 2).sum()),
            })
        chart_df = pd.DataFrame(data)
        melted = chart_df.melt('Критерий', var_name='Тип свидетельства', value_name='Отчётов')
        fig = px.bar(
            melted,
            x='Отчётов',
            y='Критерий',
            color='Тип свидетельства',
            orientation='h',
            title='Охват 16 критериев анкеты в отчётах наставников',
            barmode='stack',
            height=640,
            color_discrete_map={
                'Упоминание / действие (1)': '#4C78A8',
                'Описанное изменение / динамика (2)': '#2ECC71',
            },
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'}, margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with b:
        dates = pd.to_datetime(filtered.date, errors='coerce')
        dated = filtered.assign(month=dates.dt.strftime('%Y-%m')).dropna(subset=['month'])
        bymonth = dated.groupby('month').size().reset_index(name='Отчётов')
        if not bymonth.empty:
            fig_m = px.line(bymonth, x='month', y='Отчётов', markers=True, title='Динамика поступления отчётов по месяцам')
            st.plotly_chart(fig_m, use_container_width=True)
        statuses = filtered.analysis_status.value_counts().rename_axis('Статус').reset_index(name='Отчётов')
        fig_s = px.bar(statuses, x='Статус', y='Отчётов', title='Полнота обработки выгрузки')
        st.plotly_chart(fig_s, use_container_width=True)

    st.divider()
    st.subheader('Обобщающие вопросы анкеты 17–19 (комплексная оценка динамики)')
    st.caption(
        'Формулировки соответствуют документу «Вопросы анкеты наставника.docx». '
        'Показатели рассчитываются как на уровне отдельных отчётов, так и по траектории каждой пары «наставник–подопечный».'
    )
    summary_rows = []
    total_pairs_filtered = max(1, filtered[['volunteer_id', 'case_id']].drop_duplicates().shape[0])
    for q_key, meta in SUMMARY_QUESTIONS.items():
        direct_s = numeric(filtered, q_key)
        any_mention = direct_s >= 1
        any_change = direct_s == 2
        for rk in meta['related_criteria']:
            rs = numeric(filtered, rk)
            any_mention |= (rs >= 1)
            any_change |= (rs == 2)
        pairs_with_dyn = filtered.loc[any_change, ['volunteer_id', 'case_id']].drop_duplicates().shape[0]
        summary_rows.append({
            'Обобщающий вопрос анкеты': meta['title'],
            'Описание из анкеты фонда': meta['question'],
            'Отчётов с работой по направлению (score≥1)': int(any_mention.sum()),
            'Отчётов с явной динамикой (score=2)': int(any_change.sum()),
            'Пар с зафиксированным прогрессом': f'{pairs_with_dyn} из {total_pairs_filtered} ({round(100*pairs_with_dyn/total_pairs_filtered, 1)}%)',
        })
    st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)

    with st.expander('Показать подтверждающие цитаты динамики по обобщающим вопросам 17–19', expanded=False):
        for q_key, meta in SUMMARY_QUESTIONS.items():
            st.markdown(f'#### {meta["title"]}')
            q_scores = numeric(filtered, q_key)
            dyn_sub = filtered[q_scores == 2]
            if dyn_sub.empty:
                st.info('В текущей выборке нет отчётов с прямой цитатой изменения (score=2) по этому обобщающему вопросу.')
            else:
                for _, r in dyn_sub.head(15).iterrows():
                    q_txt = maybe_mask_text(str(r.get(f'{q_key}_quote', '')), mask_names_ui)
                    st.markdown(f'- **{r.get("volunteer_id")} · {r.get("case_id")} · {r.get("date")}**: «{q_txt}»')

with t_flags:
    st.subheader('Детектор «красных флагов» (3 категории риска)')
    st.caption(
        'Алгоритм находит предварительные сигналы для проверки координатором и не заменяет профессиональную супервизию. '
        'Согласно регламенту, красный флаг выгорания формируется ТОЛЬКО при наличии маркеров в ≥2 последовательных отчётах одной пары.'
    )
    if 'burnout_order_uncertain' in filtered and bools(filtered, 'burnout_order_uncertain').any():
        uncertain_cases = filtered.loc[bools(filtered, 'burnout_order_uncertain'), 'case_id'].unique().tolist()
        st.warning(
            f'Для {len(uncertain_cases)} кейсов ({", ".join(uncertain_cases[:6])}) есть несколько отчётов за одну дату: '
            'автоматическое построение серии выгорания для них заблокировано до ручной проверки хронологии.'
        )
    for kind, title in [
        ('burnout', '1. Выгорание волонтёра (серии ≥2 последовательных отчётов одной пары по 10 маркерам)'),
        ('danger', '2. Опасность для жизни и здоровья волонтёра или подопечного'),
        ('rules', '3. Нарушение волонтёром правил проекта'),
    ]:
        st.markdown('#### ' + title)
        if kind == 'rules' and ('rules_configured' not in df or not bools(df, 'rules_configured').any()):
            st.warning(
                'Регламент нарушений (`rules_policy.txt`) сейчас пуст — фонд не передал перечень дисциплинарных правил. '
                'Вы можете задать правила на вкладке «⚙️ Настройка критериев».'
            )
        current = [item for item in all_signals if item['kind'] == kind]
        if not current:
            st.info('Автоматических сигналов по выбранным фильтрам не обнаружено.')
        for item in current[:100]:
            review = reviews.get(item['signal_id'], {})
            status_badge = review.get('status', 'Не проверено')
            with st.expander(
                f'[{status_badge}] Волонтёр {item["volunteer_id"]} · {item["case_id"]} · {item["date"]} · строка Excel {item["source_row"]}',
                expanded=(kind == 'burnout'),
            ):
                st.markdown('**Дословная цитата / выдержки серии:**')
                st.write(maybe_mask_text(item['quote'], mask_names_ui))
                if item['reason']:
                    st.caption('Пояснение анализатора: ' + item['reason'])
                if review.get('comment'):
                    st.info('Комментарий координатора: ' + review['comment'])
                st.caption('Вынести или изменить решение можно на вкладке «✅ Проверка координатора».')

    st.divider()
    st.markdown('#### Мониторинг одиночных признаков выгорания (1 отчёт без серии — ещё не красный флаг)')
    singles = filtered[bools(filtered, 'burnout_marker_detected') & ~bools(filtered, 'burnout_detected')].copy()
    st.write(f'Найдено одиночных наблюдений эмоционального напряжения наставника: **{len(singles)}**')
    if not singles.empty:
        disp_singles = singles[[c for c in ['date', 'volunteer_id', 'case_id', 'source_row', 'burnout_markers', 'burnout_quote', 'burnout_reason'] if c in singles]].copy()
        if mask_names_ui and 'burnout_quote' in disp_singles:
            disp_singles['burnout_quote'] = disp_singles['burnout_quote'].map(lambda s: maybe_mask_text(s, True))
        st.dataframe(disp_singles, hide_index=True, use_container_width=True)

    with st.expander('Справочник 10 категорий эмоционального неблагополучия наставника', expanded=False):
        st.table(pd.DataFrame([{'№': k, 'Категория риска выгорания': v} for k, v in BURNOUT_MARKERS.items()]))

with t_one:
    st.subheader('Критерии 1–5: Личностное и социальное развитие')
    only_dyn_1 = st.checkbox('Показывать только отчёты с явно описанной динамикой (score=2)', key='dyn_1')
    render_criteria(CRITERIA_GROUPS[0][1], filtered, only_dynamics=only_dyn_1, mask_names=mask_names_ui)

with t_two:
    st.subheader('Критерии 6–9 и 16: Практические навыки, учёба, работа и будущее')
    only_dyn_2 = st.checkbox('Показывать только отчёты с явно описанной динамикой (score=2)', key='dyn_2')
    render_criteria(CRITERIA_GROUPS[1][1], filtered, only_dynamics=only_dyn_2, mask_names=mask_names_ui)

with t_three:
    st.subheader('Критерии 10–15: Здоровье, финансы, правовая грамотность и безопасность')
    only_dyn_3 = st.checkbox('Показывать только отчёты с явно описанной динамикой (score=2)', key='dyn_3')
    render_criteria(CRITERIA_GROUPS[2][1], filtered, only_dynamics=only_dyn_3, mask_names=mask_names_ui)

with t_extra:
    extras = {i['key']: i['description'] for i in cfg if not i['core'] and i['enabled']}
    if extras:
        st.subheader('Дополнительные критерии координатора')
        render_criteria(extras, filtered, mask_names=mask_names_ui)
        st.divider()

    st.subheader('Поиск неучтённых тем: кластеризация и предложение новых критериев')
    st.caption(
        'Система анализирует предложения из отчётов, не покрытые 16 базовыми критериями, '
        'группирует их в тематические кластеры и предлагает готовые формулировки новых критериев для координатора.'
    )
    topic_clusters = discover_unaccounted_topics(filtered, max_clusters=6)
    if not topic_clusters:
        st.info('В текущей выборке недостаточно данных для выделения новых кластеров.')
    else:
        existing_titles = {x['title'].strip().lower() for x in cfg}
        for cl in topic_clusters:
            with st.expander(
                f'📌 {cl["title"]} — {cl["report_count"]} отчётов ({cl["share_pct"]}% выборки)',
                expanded=True,
            ):
                st.markdown(f'**Предлагаемая инструкция для анализатора:** {cl["description"]}')
                st.caption(f'Ключевые маркеры кластера: `{", ".join(cl["keywords"])}` · Метод: {cl["method"]}')
                if cl['examples']:
                    st.markdown('**Примеры из реальных отчётов:**')
                    for ex in cl['examples']:
                        q_disp = maybe_mask_text(ex['quote'], mask_names_ui)
                        st.markdown(f'- *{ex["date"]} · Волонтёр {ex["volunteer_id"]} ({ex["case_id"]}, стр. {ex["source_row"]})*: «{q_disp}»')
                already_added = cl['title'].strip().lower() in existing_titles
                if st.button(
                    '✅ Уже добавлен в критерии' if already_added else '➕ Добавить как новый критерий анализа',
                    key=f'add_cluster_{cl["cluster_id"]}',
                    disabled=already_added,
                ):
                    seq = 1
                    taken = {x['key'] for x in cfg}
                    while f'custom_{seq:03d}' in taken:
                        seq += 1
                    cfg.append({
                        'key': f'custom_{seq:03d}',
                        'title': cl['title'],
                        'description': cl['description'],
                        'enabled': True,
                        'core': False,
                    })
                    save_config(cfg)
                    st.success(f'Критерий «{cl["title"]}» добавлен! Нажмите «▶ Запустить анализ отчётов» в боковой панели для расчёта оценок.')
                    st.rerun()

    st.divider()
    with st.expander('Частотный словарь слов в отчётах без найденных категорий', expanded=False):
        words = suggested_words(filtered)
        if words:
            st.dataframe(pd.DataFrame(words, columns=['Слово', 'Отчётов']), hide_index=True, use_container_width=True)
        else:
            st.info('Все отчёты выборки покрыты хотя бы одним критерием.')

with t_history:
    st.subheader('Полные отчёты и хронологическая история пар')
    if filtered.empty:
        st.info('Нет записей.')
    else:
        order = filtered.copy()
        order['_date'] = pd.to_datetime(order.date, errors='coerce')
        order = order.sort_values(['volunteer_id', 'case_id', '_date', 'source_row'], ascending=[True, True, False, False])
        for _, r in order.head(160).iterrows():
            active_tags = []
            for k in all_active:
                sc = pd.to_numeric(r.get(f'{k}_score'), errors='coerce')
                if sc == 2:
                    active_tags.append(f'🟢 {k}=2')
                elif sc == 1:
                    active_tags.append(f'🔵 {k}=1')
            if as_bool(r.get('burnout_detected')):
                active_tags.append('🚩 СЕРИЯ ВЫГОРАНИЯ')
            elif as_bool(r.get('burnout_marker_detected')):
                active_tags.append('⚠️ одиночный маркер выгорания')
            if as_bool(r.get('danger_detected')):
                active_tags.append('🚨 ОПАСНОСТЬ')

            tags_str = (' · ' + ', '.join(active_tags[:6])) if active_tags else ''
            header = (
                f'{r["date"]} · Волонтёр {r["volunteer_id"]} · {r["case_id"]} · '
                f'строка Excel {r.get("source_row", "?")} · {r.get("analysis_status", "")}{tags_str}'
            )
            with st.expander(header):
                raw_txt = str(r['text']).strip()
                if not raw_txt:
                    st.write('(пустой отчёт)')
                else:
                    st.write(maybe_mask_text(raw_txt, mask_names_ui))
                if str(r.get('analysis_errors', '')).strip():
                    st.caption('Ошибки анализа: ' + str(r['analysis_errors']))
        if len(order) > 160:
            st.caption('Показаны первые 160 записей. Для остальных используйте фильтры.')

with t_reviews:
    st.subheader('Проверка и подтверждение сигналов координатором')
    if not all_signals:
        st.info('Нет красных флагов в выборке. Одиночные маркеры выгорания не требуют обязательного красного предупреждения.')
    else:
        options = {
            f'[{reviews.get(item["signal_id"], {}).get("status", "Не проверено")}] '
            f'{item["category"]} · Волонтёр {item["volunteer_id"]} · {item["case_id"]} · {item["date"]} · ID {item["signal_id"][:6]}': item
            for item in all_signals
        }
        chosen = st.selectbox('Выберите сигнал на проверку', list(options))
        selected = options[chosen]
        st.markdown('**Исходная цитата сигнала:**')
        st.write(maybe_mask_text(selected['quote'], mask_names_ui))
        if selected.get('reason'):
            st.caption('Обоснование: ' + selected['reason'])
        last = reviews.get(selected['signal_id'], {})
        with st.form('review_form'):
            status = st.selectbox('Решение координатора', STATUSES, index=STATUSES.index(last.get('status', STATUSES[0])))
            comment = st.text_area('Комментарий координатора (сохраняется для выгрузки в 1С CRM)', value=last.get('comment', ''), height=100)
            submit = st.form_submit_button('Сохранить решение в SQLite')
        if submit:
            set_review(selected['signal_id'], status, comment)
            st.success('Решение сохранено в локальном журнале координатора.')
            st.rerun()

    st.divider()
    st.markdown('#### Сводный журнал проверенных сигналов')
    st.dataframe(signal_export(filtered, reviews), use_container_width=True, hide_index=True)

with t_settings:
    st.subheader('Гибкая настройка критериев и правил без программирования')
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown('#### Редактировать существующий критерий')
        selected_key = st.selectbox(
            'Выберите критерий',
            [i['key'] for i in cfg],
            format_func=lambda k: next(f'{i["title"]} ({k})' for i in cfg if i['key'] == k),
        )
        item = next(i for i in cfg if i['key'] == selected_key)
        with st.form('edit_criterion'):
            title = st.text_input('Название', value=item['title'])
            description = st.text_area('Инструкция для модели / ключевые ориентиры', value=item['description'], height=110)
            enabled = st.checkbox('Критерий включён', value=item['enabled'], disabled=item['core'])
            submit = st.form_submit_button('Сохранить изменения критерия')
        if submit:
            item.update(title=title.strip(), description=description.strip(), enabled=enabled or item['core'])
            try:
                save_config(cfg)
                st.success('Сохранено. Запустите анализ в боковой панели для пересчёта оценок.')
            except Exception as exc:
                st.error(str(exc))

        st.markdown('#### Добавить новый критерий')
        with st.form('new_criterion'):
            new_title = st.text_input('Название нового критерия')
            new_description = st.text_area('Что должна искать система в отчётах')
            add = st.form_submit_button('Добавить критерий')
        if add:
            if not new_title.strip() or not new_description.strip():
                st.error('Нужны название и описание')
            else:
                seq = 1
                taken = {x['key'] for x in cfg}
                while f'custom_{seq:03d}' in taken:
                    seq += 1
                cfg.append(dict(
                    key=f'custom_{seq:03d}',
                    title=new_title.strip(),
                    description=new_description.strip(),
                    enabled=True,
                    core=False,
                ))
                save_config(cfg)
                st.success('Дополнительный критерий сохранён! Запустите анализ для заполнения его оценок.')
                st.rerun()

    with col_c2:
        st.markdown('#### Регламент правил проекта для детектора нарушений (`rules_policy.txt`)')
        st.caption(
            'По умолчанию файл пуст, так как официальный регламент не был приложен к кейсу. '
            'Вы можете внести сюда правила фонда (или загрузить типовой шаблон), чтобы включить автоматическую проверку.'
        )
        current_policy = POLICY_FILE.read_text(encoding='utf-8') if POLICY_FILE.exists() else ''
        if st.button('Заполнить типовым шаблоном правил наставничества'):
            current_policy = (
                '1. Запрещено самостоятельно увозить подопечного к кровным родственникам без письменного согласования с координатором и администрацией ЦССВ.\n'
                '2. Запрещено оставлять подопечного на ночёвку без оформленного гостевого режима.\n'
                '3. Запрещено передавать подопечному крупные суммы наличных денег, оформлять на себя кредиты, дроны или сим-карты без согласования.\n'
                '4. Запрещено публиковать фотографии и персональные данные подопечного в открытых источниках.'
            )
            POLICY_FILE.write_text(current_policy, encoding='utf-8')
            st.success('Типовой шаблон правил записан в rules_policy.txt.')
            st.rerun()
        with st.form('policy_form'):
            policy_text = st.text_area('Текст правил фонда (оставьте пустым для отключения детектора нарушений)', value=current_policy, height=220)
            save_pol = st.form_submit_button('Сохранить регламент правил')
        if save_pol:
            POLICY_FILE.write_text(policy_text.strip(), encoding='utf-8')
            st.success('Файл rules_policy.txt обновлён. Перезапустите анализ для проверки отчётов по новым правилам.')

with t_quality:
    st.subheader('Пайплайн проверки качества модели и Обратимая псевдонимизация лиц')
    q_col1, q_col2 = st.columns(2)
    with q_col1:
        st.markdown('#### 1. Автоматический эталонный бенчмарк (Gold Benchmark)')
        st.caption(
            'Позволяет при смене модели или провайдера за секунды проверить точность классификации 16 критериев, '
            'различение упоминания (1) и динамики (2), дословность цитат и защиту от ложных срабатываний, когда устал ребёнок, а не наставник.'
        )
        bm_provider = st.selectbox(
            'Провайдер для тестирования',
            ['offline', 'auto', 'ollama', 'openai'],
            key='bm_prov',
        )
        if st.button('▶ Запустить проверку на эталонном наборе (10 сценариев)'):
            try:
                bm_df, bm_sum = run_benchmark(provider=bm_provider)
                k1, k2, k3, k4 = st.columns(4)
                k1.metric('F1 по критериям', f'{bm_sum["criteria_f1"]:.3f}')
                k2.metric('Точность 0/1/2', f'{bm_sum["score_exact_accuracy"]*100:.0f}%')
                k3.metric('Дословность цитат', f'{bm_sum["quote_faithfulness"]*100:.0f}%')
                k4.metric('Ловушки субъекта', f'{bm_sum["subject_trap_pass_rate"]*100:.0f}%')
                st.dataframe(bm_df, hide_index=True, use_container_width=True)
            except Exception as exc:
                st.error(f'Ошибка при запуске бенчмарка: {exc}')

        st.divider()
        st.markdown('#### Ручная валидация на выборке отчётов (`gold_template_PRIVATE.csv`)')
        if st.button('Сформировать выборку из 30 отчётов для ручной разметки'):
            out_gold = HERE / 'gold_template_PRIVATE.csv'
            create_sample(RESULTS, out_gold, n=30)
            st.success('Файл gold_template_PRIVATE.csv создан.')
        if (HERE / 'gold_template_PRIVATE.csv').exists():
            st.download_button(
                'Скачать шаблон ручной разметки (gold_template_PRIVATE.csv)',
                (HERE / 'gold_template_PRIVATE.csv').read_bytes(),
                file_name='gold_template_PRIVATE.csv',
                mime='text/csv',
            )
        uploaded_gold = st.file_uploader('Загрузить размеченный файл gold_template_PRIVATE.csv для расчёта метрик', type=['csv'])
        if uploaded_gold is not None:
            gold_df = pd.read_csv(uploaded_gold, keep_default_na=False)
            eval_res = evaluate_dataframe(gold_df)
            if eval_res.empty:
                st.warning('В загруженном файле ещё не заполнены столбцы gold_... значениями 0 или 1.')
            else:
                st.dataframe(eval_res, hide_index=True, use_container_width=True)

    with q_col2:
        st.markdown('#### 2. Обратимая автоматическая псевдонимизация лиц')
        st.caption(
            'Автоматически распознаёт русские имена (включая уменьшительные формы и все 6 падежей: '
            '*Ваня/Вани/Ваней*, *Маша/Машей/Машу*), отчества и фамилии (*Кристина Панкова*, *Наталья Сергеевна*), '
            'группирует формы одного человека в единый токен `[ЛИЦО_N]` и гарантирует 100% точное обратное восстановление.'
        )
        sample_rows = filtered[filtered['text'].astype(str).str.len() > 80].head(30)
        sample_options = {
            f'Строка {r.source_row} · {r.case_id} · {r.date}': str(r.text)
            for _, r in sample_rows.iterrows()
        }
        default_sample = next(iter(sample_options.values()), 'Сегодня гуляли с Ваней и Инной, встретили Наталью Сергеевну. У Вани отличное настроение!')
        chosen_sample_label = st.selectbox('Выбрать реальный отчёт для демонстрации', ['Свой текст'] + list(sample_options.keys()))
        init_text = default_sample if chosen_sample_label == 'Свой текст' else sample_options[chosen_sample_label]

        personal_text = st.text_area('Текст отчёта для псевдонимизации', value=init_text, height=130)
        extra_names_str = st.text_input('Дополнительные имена/обозначения через запятую (необязательно)', value='')
        extra_list = [v.strip() for v in extra_names_str.split(',') if v.strip()]

        masked_text, mapping_dict, entities_info = pseudonymize_detailed(personal_text, extra_names=extra_list, auto_detect=True)
        restored_text = restore(masked_text, mapping_dict)

        st.markdown('**Обезличенный текст (передаётся в модель):**')
        st.code(masked_text, language=None)
        if entities_info:
            st.markdown('**Обнаруженные лица и падежные формы (ключ координатора):**')
            st.dataframe(
                pd.DataFrame([
                    {
                        'Токен': x['person_id'],
                        'Каноническое имя': x['canonical_name'],
                        'Формы в тексте': ', '.join(x['forms']),
                        'Упоминаний': x['count'],
                    }
                    for x in entities_info
                ]),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info('Имена собственные в данном тексте не обнаружены.')

        exact_ok = (restored_text == personal_text)
        st.success(f'Проверка обратного восстановления: {"100% точное совпадение с оригиналом ✅" if exact_ok else "Ошибка ❌"}')

with t_export:
    st.subheader('Экспорт результатов для CRM фонда (на базе 1С)')
    st.caption(
        'Формирует готовые таблицы в форматах CSV (UTF-8 с BOM) и Excel (.xlsx) для загрузки в 1С CRM: '
        'сводную ведомость по каждому волонтёру/кейсу (замена квартальной анкеты) и реестр подтверждённых сигналов риска.'
    )

    st.markdown('#### 1. Сводная ведомость по парам «волонтёр + кейс» (16 критериев + вопросы 17–19 + риски)')
    crm_summary_df = crm_volunteer_summary_export(filtered, reviews, cfg)
    st.dataframe(crm_summary_df.head(50), use_container_width=True, hide_index=True)
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            'Скачать сводную ведомость по волонтёрам (CSV для 1С)',
            safe_csv(crm_summary_df),
            'crm_1c_volunteers_summary.csv',
            'text/csv',
        )
    with dl_col2:
        st.download_button(
            'Скачать сводную ведомость по волонтёрам (Excel .xlsx для 1С)',
            safe_excel(crm_summary_df, sheet_name='Волонтеры_1С'),
            'crm_1c_volunteers_summary.xlsx',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        )

    st.divider()
    st.markdown('#### 2. Выгрузка сигналов риска и решений координатора для 1С CRM')
    export_signals = signal_export(filtered, reviews)
    only_confirmed = st.checkbox('Экспортировать только подтверждённые координатором сигналы («Подтверждено»)', value=False)
    if only_confirmed:
        export_signals = export_signals[export_signals.review_status == 'Подтверждено']
    st.dataframe(export_signals, use_container_width=True, hide_index=True)
    s_col1, s_col2 = st.columns(2)
    with s_col1:
        st.download_button(
            '📥 Скачать сигналы и решения (CSV для 1С CRM)',
            safe_csv(export_signals),
            'crm_signals.csv',
            'text/csv',
        )
    with s_col2:
        st.download_button(
            '📥 Скачать сигналы и решения (Excel .xlsx для 1С CRM)',
            safe_excel(export_signals, sheet_name='Сигналы_1С'),
            'crm_signals.xlsx',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        )

    st.divider()
    st.markdown('#### 3. Детальный экспорт обработанных отчётов')
    pseudo_export = st.checkbox('Автоматически обезличить все имена в текстах и цитатах при выгрузке', value=True)
    if st.checkbox('Показать технический экспорт всех строк отчётов', value=False):
        export_full = filtered.copy()
        if pseudo_export:
            text_like_cols = ['text'] + [c for c in export_full.columns if c.endswith('_quote')]
            for col in text_like_cols:
                if col in export_full.columns:
                    export_full[col] = export_full[col].map(lambda s: maybe_mask_text(s, True))
        else:
            st.warning('Внимание: выгрузка содержит необезличенные тексты отчётов. Храните файл в защищённом контуре.')
        st.download_button(
            'Скачать полный детальный CSV',
            safe_csv(export_full),
            'full_analysis_pseudonymized.csv' if pseudo_export else 'full_analysis_PRIVATE.csv',
            'text/csv',
        )

st.sidebar.divider()
st.sidebar.download_button(
    'Сводка по фильтру для 1С (CSV)',
    safe_csv(crm_volunteer_summary_export(filtered, reviews, cfg)),
    file_name='selected_volunteers_1c.csv',
    mime='text/csv',
)
st.sidebar.caption('Все вычисления выполняются локально. Таблица соответствий псевдонимов хранится отдельно от модельных промптов.')
