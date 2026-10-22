"""Human confirmation of candidate signals, saved to local SQLite; plus 1C CRM exports."""
from __future__ import annotations
import hashlib
import sqlite3
from pathlib import Path
import pandas as pd

from criteria import ALL_CRITERIA, SUMMARY_QUESTIONS

HERE = Path(__file__).resolve().parent
DB = HERE / 'coordinator_reviews.sqlite3'
STATUSES = ('Не проверено', 'Подтверждено', 'Отклонено')


def as_bool(v):
    return v is True or str(v).lower().strip() in ('true', '1', 'yes')


def signal_id(kind, volunteer, case, source_row, quote):
    raw = '|'.join(map(str, (kind, volunteer, case, source_row, quote)))
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()[:20]


def signals(df):
    """One review object per burnout sequence; individual objects for other risks."""
    result = []
    if df.empty:
        return result
    for _, record in df.iterrows():
        r = record.to_dict()
        for kind, detected, quote, label in (
            ('burnout', 'burnout_detected', 'burnout_quote', 'Риск выгорания'),
            ('danger', 'danger_detected', 'danger_quote', 'Опасность жизни и здоровью'),
            ('rules', 'rules_detected', 'rules_quote', 'Возможное нарушение правил'),
        ):
            if not as_bool(r.get(detected, False)):
                continue
            seq = str(r.get('burnout_sequence_id', '')).strip() if kind == 'burnout' else ''
            identity = seq if seq else str(r.get('source_row', ''))
            quote_text = str(r.get(quote, '')).strip()
            uid = signal_id(kind, r.get('volunteer_id', ''), r.get('case_id', ''), identity,
                            '' if seq else quote_text)
            result.append(dict(
                signal_id=uid,
                kind=kind,
                category=label,
                volunteer_id=str(r.get('volunteer_id', '')),
                case_id=str(r.get('case_id', '')),
                source_row=str(r.get('source_row', '')),
                date=str(r.get('date', '')),
                quote=f'{r.get("date", "")}: {quote_text}' if kind == 'burnout' and quote_text else quote_text,
                sequence_id=seq,
                reason=str(r.get(kind + '_reason', '')),
            ))
    by_id = {}
    for item in result:
        if item['signal_id'] not in by_id:
            by_id[item['signal_id']] = item
        elif item['kind'] == 'burnout':
            old = by_id[item['signal_id']]
            old['quote'] += '\n' + item['quote']
            if item['date'] > old['date']:
                old['date'] = item['date']
    return list(by_id.values())


def _connect(db=DB):
    connection = sqlite3.connect(db)
    connection.execute('''CREATE TABLE IF NOT EXISTS reviews
        (signal_id TEXT PRIMARY KEY, status TEXT NOT NULL, comment TEXT NOT NULL,
         reviewed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)''')
    return connection


def list_reviews(db=DB):
    with _connect(db) as conn:
        return {i: dict(status=s, comment=c, reviewed_at=a)
                for i, s, c, a in conn.execute('SELECT signal_id,status,comment,reviewed_at FROM reviews')}


def set_review(signal_id, status, comment='', db=DB):
    if status not in STATUSES:
        raise ValueError('Недопустимый статус проверки')
    with _connect(db) as conn:
        conn.execute('''INSERT INTO reviews(signal_id,status,comment) VALUES(?,?,?)
            ON CONFLICT(signal_id) DO UPDATE SET status=excluded.status,
            comment=excluded.comment,reviewed_at=CURRENT_TIMESTAMP''',
                     (signal_id, status, str(comment)[:4000]))


def signal_export(df, reviews=None):
    reviews = reviews if reviews is not None else list_reviews()
    rows = []
    for item in signals(df):
        decision = reviews.get(item['signal_id'], {})
        rows.append({**item, 'review_status': decision.get('status', 'Не проверено'),
                     'review_comment': decision.get('comment', ''),
                     'reviewed_at': decision.get('reviewed_at', '')})
    return pd.DataFrame(rows, columns=['signal_id', 'category', 'kind', 'volunteer_id',
            'case_id', 'source_row', 'date', 'quote', 'reason', 'sequence_id',
            'review_status', 'review_comment', 'reviewed_at'])


def _num_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(0, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors='coerce').fillna(0)


def crm_volunteer_summary_export(df: pd.DataFrame, reviews: dict | None = None, criteria_cfg: list[dict] | None = None) -> pd.DataFrame:
    """Формирует сводную таблицу по парам (волонтёр + кейс) для загрузки в CRM фонда на базе 1С.

    Заменяет квартальное гугл-анкетирование: агрегирует по каждой паре количество отчётов,
    период наблюдений, упоминания и зафиксированную динамику (score=2) по всем 16+ критериям,
    обобщающим вопросам 17–19 и подтверждённым/предварительным сигналам риска.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    reviews = reviews if reviews is not None else list_reviews()
    all_sigs = signals(df)

    # Определяем список активных критериев
    if criteria_cfg:
        crit_keys = [(c['key'], c['title']) for c in criteria_cfg if c.get('enabled', True)]
    else:
        crit_keys = list(ALL_CRITERIA.items())

    rows = []
    for (vol, case), grp in df.groupby(['volunteer_id', 'case_id'], dropna=False):
        dates = pd.to_datetime(grp['date'], errors='coerce').dropna()
        p_start = dates.min().strftime('%Y-%m-%d') if not dates.empty else ''
        p_end = dates.max().strftime('%Y-%m-%d') if not dates.empty else ''
        total_rep = len(grp)
        ok_rep = int(grp['analysis_status'].astype(str).eq('ok').sum()) if 'analysis_status' in grp else total_rep

        pair_sigs = [s for s in all_sigs if s['volunteer_id'] == str(vol) and s['case_id'] == str(case)]
        n_burnout_seq = sum(1 for s in pair_sigs if s['kind'] == 'burnout')
        n_danger = sum(1 for s in pair_sigs if s['kind'] == 'danger')
        n_rules = sum(1 for s in pair_sigs if s['kind'] == 'rules')
        n_confirmed = sum(1 for s in pair_sigs if reviews.get(s['signal_id'], {}).get('status') == 'Подтверждено')
        n_single_burnout = int(
            (grp['burnout_marker_detected'].map(as_bool) & ~grp['burnout_detected'].map(as_bool)).sum()
        ) if 'burnout_marker_detected' in grp and 'burnout_detected' in grp else 0

        # Обобщающие вопросы 17–19
        q_summary = {}
        for q_key, meta in SUMMARY_QUESTIONS.items():
            # Проверяем либо прямой столбец q_key_score, либо связанные критерии
            direct = _num_series(grp, f'{q_key}_score')
            related_max = pd.Series(0.0, index=grp.index)
            for rk in meta['related_criteria']:
                related_max = related_max.combine(_num_series(grp, f'{rk}_score'), max)
            combined = direct.combine(related_max, max)
            n_mentions = int((combined >= 1).sum())
            n_dyn = int((combined == 2).sum())
            if n_dyn >= 2:
                level = 'Устойчивая положительная динамика (2+ отчётов с изменениями)'
            elif n_dyn == 1:
                level = 'Зафиксированы отдельные изменения (1 отчёт с динамикой)'
            elif n_mentions >= 1:
                level = 'Ведётся работа без явно описанных изменений'
            else:
                level = 'Нет сведений в отчётах периода'
            q_summary[f'q{meta["number"]}_status'] = level
            q_summary[f'q{meta["number"]}_mentions'] = n_mentions
            q_summary[f'q{meta["number"]}_dynamics'] = n_dyn

        row_dict = {
            'volunteer_id': str(vol),
            'case_id': str(case),
            'period_start': p_start,
            'period_end': p_end,
            'reports_total': total_rep,
            'reports_analyzed_ok': ok_rep,
            'q17_wellbeing_status': q_summary['q17_status'],
            'q17_wellbeing_dynamics_reports': q_summary['q17_dynamics'],
            'q18_socialization_status': q_summary['q18_status'],
            'q18_socialization_dynamics_reports': q_summary['q18_dynamics'],
            'q19_autonomy_status': q_summary['q19_status'],
            'q19_autonomy_dynamics_reports': q_summary['q19_dynamics'],
            'burnout_series_signals': n_burnout_seq,
            'burnout_single_markers': n_single_burnout,
            'danger_signals': n_danger,
            'rules_signals': n_rules,
            'coordinator_confirmed_signals': n_confirmed,
        }

        total_criteria_covered = 0
        total_dynamics_events = 0
        for c_key, _ in crit_keys:
            scores = _num_series(grp, f'{c_key}_score')
            mentions = int((scores >= 1).sum())
            dynamics = int((scores == 2).sum())
            if mentions > 0:
                total_criteria_covered += 1
            total_dynamics_events += dynamics
            row_dict[f'{c_key}_mentions'] = mentions
            row_dict[f'{c_key}_dynamics'] = dynamics
            # Последняя подтверждающая цитата для карточки 1С
            best_q = ''
            if f'{c_key}_quote' in grp.columns:
                sub_q = grp.loc[scores >= 1].sort_values('date', ascending=False)
                for _, qr in sub_q.iterrows():
                    cand_q = str(qr.get(f'{c_key}_quote', '')).strip()
                    if cand_q:
                        best_q = f'{qr.get("date", "")}: {cand_q[:180]}'
                        break
            row_dict[f'{c_key}_latest_quote'] = best_q

        row_dict['criteria_covered_count'] = total_criteria_covered
        row_dict['total_dynamics_mentions'] = total_dynamics_events
        rows.append(row_dict)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(['reports_total', 'total_dynamics_mentions'], ascending=[False, False]).reset_index(drop=True)
    return out
