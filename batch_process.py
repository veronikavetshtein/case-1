from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd

from burnout_rules import add_burnout_sequences
from criteria import SUMMARY_QUESTIONS
from criteria_store import active_criteria, criteria_groups, config_fingerprint
from llm_client import LocalLLM, ModelError
from parser import parse_excel
from pseudonymize import pseudonymize_detailed, save_private_map

HERE = Path(__file__).resolve().parent


def make_blank(row, keys=None):
    keys = keys if keys is not None else active_criteria()
    item = {k: row.get(k, '') for k in ('source_row', 'case_id', 'volunteer_id', 'date', 'text', 'date_raw')}
    item.update({
        'analysis_status': 'unprocessed',
        'analysis_errors': '',
        'red_flags_status': 'unprocessed',
        'criteria_version': '',
        'burnout_marker_detected': False,
        'burnout_markers': '[]',
        'burnout_quote': '',
        'burnout_reason': '',
        'danger_detected': False,
        'danger_quote': '',
        'danger_reason': '',
        'rules_detected': False,
        'rules_quote': '',
        'rules_reason': '',
        'rules_configured': False,
    })
    for key in keys:
        item[f'{key}_score'] = None
        item[f'{key}_quote'] = ''
    for q_key in SUMMARY_QUESTIONS:
        item[f'{q_key}_score'] = None
        item[f'{q_key}_quote'] = ''
    return item


def valid_answer(value):
    return isinstance(value, dict) and type(value.get('score')) is int and value['score'] in (0, 1, 2)


def analyze_one(row, llm, fast=False, criteria=None, groups=None, prior=None):
    criteria = criteria if criteria is not None else active_criteria()
    groups = groups if groups is not None else criteria_groups()
    result = make_blank(row, criteria)
    report = str(row.get('text', '')).strip()
    if not report:
        result['analysis_status'] = 'empty'
        result['red_flags_status'] = 'empty'
        return result
    errors = []
    found = {}
    prior = prior or {}
    # Preserve previously validated scores/quotes for partial reports.
    for key in criteria:
        old_score = str(prior.get(key + '_score', '')).strip()
        old_quote = str(prior.get(key + '_quote', '')).strip()
        try:
            number = float(old_score)
        except (TypeError, ValueError):
            continue
        if number not in (0., 1., 2.):
            continue
        score = int(number)
        if score != 0 and (not old_quote or old_quote not in report):
            continue
        found[key] = {'score': score, 'quote': '' if score == 0 else old_quote}
    # Fast attempt once on 16+ keys; if incomplete, retry only the missing keys.
    missing = [k for k in criteria if k not in found]
    if fast and len(missing) > 6:
        fast_subset = {k: criteria[k] for k in missing}
        response = llm.analyze_criteria_group(report, fast_subset, 'Все недостающие критерии')
        found.update({k: v for k, v in response.items() if k in fast_subset and valid_answer(v)})
        missing = [k for k in criteria if k not in found]
        if missing:
            print(f'  [FAST] Модель вернула {len(found)}/{len(criteria)} критериев; '
                  f'допроверяю только {len(missing)} недостающих', flush=True)
        elif response.get('_errors'):
            errors.extend(response['_errors'])
    for group_name, group in groups:
        subset = {k: v for k, v in group.items() if k in missing}
        if not subset:
            continue
        answer = llm.analyze_criteria_group(report, subset, group_name)
        errors.extend(answer.get('_errors', []))
        found.update({k: v for k, v in answer.items() if k in subset and valid_answer(v)})
    for key in criteria:
        val = found.get(key)
        if val is not None:
            result[key + '_score'] = val['score']
            result[key + '_quote'] = val.get('quote', '')
        else:
            errors.append(f'{key}: оценка не получена')

    # Вычисляем обобщающие вопросы 17–19 на основе проверенных оценок и цитат
    summary_ans = llm.analyze_summary_questions(report, found)
    for q_key, q_val in summary_ans.items():
        result[f'{q_key}_score'] = q_val['score']
        result[f'{q_key}_quote'] = q_val['quote']

    if str(prior.get('red_flags_status', '')) == 'ok':
        for key in ('burnout_marker_detected', 'burnout_markers', 'burnout_quote', 'burnout_reason',
                    'danger_detected', 'danger_quote', 'danger_reason',
                    'rules_detected', 'rules_quote', 'rules_reason', 'rules_configured'):
            if key in prior:
                result[key] = prior[key]
        result['red_flags_status'] = 'ok'
    else:
        flags = llm.analyze_red_flags(report)
        flag_errors = flags.get('_errors', [])
        errors.extend(flag_errors)
        result['red_flags_status'] = 'partial' if flag_errors else 'ok'
        for kind in ('burnout', 'danger', 'rules'):
            val = flags.get(kind, {})
            stem = 'burnout_marker' if kind == 'burnout' else kind
            result[f'{stem}_detected'] = val.get('detected', False)
            result[f'{kind}_quote'] = val.get('quote', '')
            result[f'{kind}_reason'] = val.get('reason', '')
            if kind == 'burnout':
                result['burnout_markers'] = json.dumps(val.get('markers', []), ensure_ascii=False)
        result['rules_configured'] = flags.get('rules_configured', False)
    result['analysis_errors'] = ' | '.join(errors)[:8000]
    result['analysis_status'] = 'partial' if errors else 'ok'
    return result


def validate_schema(df, criteria=None):
    criteria = criteria if criteria is not None else active_criteria()
    needed = ['source_row', 'case_id', 'volunteer_id', 'date', 'text', 'analysis_status',
              'burnout_marker_detected', 'burnout_detected', 'danger_detected', 'rules_detected']
    needed += [f'{k}_{suffix}' for k in criteria for suffix in ('score', 'quote')]
    missing = [x for x in needed if x not in df.columns]
    if missing:
        raise ValueError('Не созданы столбцы: ' + ', '.join(missing))
    return df


def merge_cached(blank, cache_row, keys):
    for key in blank:
        if key in cache_row:
            blank[key] = cache_row[key]
    if not str(blank.get('red_flags_status', '')).strip() or blank['red_flags_status'] == 'unprocessed':
        blank['red_flags_status'] = 'ok' if blank.get('analysis_status') == 'ok' else 'partial'
    return blank


def run_batch_analysis(
    excel_path: str | Path = 'reports.xlsx',
    output_path: str | Path = 'demo_results.csv',
    max_cases: int | None = None,
    resume: bool = False,
    fast: bool = False,
    provider: str | None = None,
    save_pseudonym_vault: bool = True,
    progress_callback=None,
) -> pd.DataFrame:
    """Программный запуск пакетного анализа (для CLI и веб-интерфейса Streamlit)."""
    excel = Path(excel_path)
    if not excel.exists():
        excel = HERE / excel_path
    if not excel.exists():
        raise FileNotFoundError(f'Не найден файл Excel: {excel_path}')

    criteria = active_criteria()
    groups = criteria_groups()
    raw = parse_excel(excel)
    source_count, empty, duplicates = len(raw), int(raw.text.eq('').sum()), int(raw.possible_duplicate.sum())
    raw = raw.loc[~raw.possible_duplicate].copy()
    if max_cases is not None:
        chosen = list(raw.case_id.drop_duplicates())[:max_cases]
        raw = raw[raw.case_id.isin(chosen)]
    records = raw.to_dict('records')
    results = [make_blank(r, criteria) for r in records]
    output = Path(output_path)
    if not output.is_absolute():
        output = HERE / output

    llm = LocalLLM(provider_override=provider)
    backend = llm._resolve_backend()
    fingerprint = config_fingerprint(f'{backend}:{llm.model}', llm.rules_policy)

    old_cache = {}
    partial_cache = {}
    if resume and output.exists():
        old = pd.read_csv(output, dtype={'volunteer_id': str, 'case_id': str}, keep_default_na=False)
        for _, row in old.iterrows():
            if str(row.get('analysis_status')) not in ('ok', 'empty', 'partial'):
                continue
            old_version = str(row.get('criteria_version', '')).strip()
            if old_version and old_version != fingerprint:
                continue
            if any(f'{key}_score' not in old for key in criteria):
                continue
            key = (str(row.get('source_row')), str(row.get('text')))
            if str(row.get('analysis_status')) == 'partial':
                partial_cache[key] = row.to_dict()
            else:
                old_cache[key] = row.to_dict()

    if records and any(str(r['text']).strip() for r in records) and len(old_cache) < len(records):
        llm.ensure_model()

    print(f'Исходных активностей: {source_count}; пустых: {empty}; возможных дублей: {duplicates}')
    print(f'Выбрано {len(records)} записей из {raw.case_id.nunique()} кейсов; '
          f'всего критериев: {len(criteria)}, провайдер={backend} ({llm.model}), fast={fast}')

    vault_entries = {}
    completed = 0
    for i, row in enumerate(records):
        cache = old_cache.get((str(row['source_row']), str(row['text'])))
        if cache:
            results[i] = merge_cached(results[i], cache, criteria)
            completed += 1
        else:
            try:
                partial = partial_cache.get((str(row['source_row']), str(row['text'])))
                results[i] = analyze_one(row, llm, fast=fast, criteria=criteria,
                                         groups=groups, prior=partial)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:
                results[i]['analysis_status'] = 'error'
                results[i]['red_flags_status'] = 'error'
                results[i]['analysis_errors'] = str(exc)[:8000]
        results[i]['criteria_version'] = fingerprint

        if save_pseudonym_vault and str(row.get('text', '')).strip():
            _, mapping, _ = pseudonymize_detailed(str(row['text']))
            if mapping:
                vault_entries[str(row['source_row'])] = mapping

        # При работе в быстром локальном режиме сохраняем на диск каждые 25 записей и в конце,
        # а при вызове внешней LLM — после каждой записи.
        should_flush = (backend != 'offline') or ((i + 1) % 25 == 0) or (i + 1 == len(records))
        if should_flush:
            interim = add_burnout_sequences(pd.DataFrame(results))
            validate_schema(interim, criteria)
            temporary = output.with_suffix('.writing.csv')
            interim.to_csv(temporary, index=False, encoding='utf-8-sig')
            temporary.replace(output)

        if progress_callback:
            progress_callback(i + 1, len(records), row, results[i]['analysis_status'])
        if backend != 'offline' or (i + 1) % 50 == 0 or (i + 1) == len(records):
            print(f'[{i + 1}/{len(records)}] {row["volunteer_id"]} {row["case_id"]}: '
                  f'{results[i]["analysis_status"]}' + (' [resume]' if cache else ''), flush=True)

    if save_pseudonym_vault and vault_entries:
        save_private_map(vault_entries)

    final_df = add_burnout_sequences(pd.DataFrame(results))
    validate_schema(final_df, criteria)
    print(f'Готово. Пропущено ранее обработанных: {completed}. Результат: {output}')
    return final_df


def main(argv=None):
    parser = argparse.ArgumentParser(description='Анализ отчётов наставников (Ollama / OpenAI API / Локальный NLP)')
    parser.add_argument('--excel', default='reports.xlsx')
    parser.add_argument('--max-cases', type=int, default=None)
    parser.add_argument('--output', default='demo_results.csv')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--fast', action='store_true', help='Сначала все критерии, затем только пропущенные')
    parser.add_argument('--provider', default=None, help='auto | ollama | openai | offline')
    args = parser.parse_args(argv)
    if args.max_cases is not None and args.max_cases < 1:
        parser.error('--max-cases должно быть >= 1')
    excel = Path(args.excel)
    if not excel.exists():
        excel = HERE / args.excel
    if not excel.exists():
        parser.error('Не найден Excel: ' + args.excel)
    run_batch_analysis(
        excel_path=excel,
        output_path=args.output,
        max_cases=args.max_cases,
        resume=args.resume,
        fast=args.fast,
        provider=args.provider,
    )


if __name__ == '__main__':
    try:
        main()
    except ModelError as exc:
        raise SystemExit(f'Ошибка запуска модели: {exc}')
