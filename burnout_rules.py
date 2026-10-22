import pandas as pd


def is_true(x):
    if x is True:
        return True
    if x is None or x is pd.NA:
        return False
    if isinstance(x, str):
        return x.strip().lower() in {'true', '1', 'yes'}
    try:
        return bool(x == 1)
    except (TypeError, ValueError):
        return False


def add_burnout_sequences(df):
    df = df.copy().reset_index(drop=True)
    df['burnout_detected'] = False
    df['burnout_streak'] = 0
    df['burnout_sequence_id'] = ''
    df['burnout_order_uncertain'] = False
    if df.empty:
        return df
    df['_parsed_date'] = pd.to_datetime(df['date'], errors='coerce')
    if 'source_row' not in df:
        df['source_row'] = range(1, len(df) + 1)
    if 'burnout_marker_detected' not in df:
        df['burnout_marker_detected'] = False
    if 'analysis_status' not in df:
        df['analysis_status'] = 'ok'

    for (vol, case), group in df.groupby(['volunteer_id', 'case_id'], dropna=False, sort=False):
        # Неизвестная дата или несколько различных отчётов в один день
        # не позволяют достоверно установить последовательность встреч.
        ambiguous = group['_parsed_date'].isna().any() or group['_parsed_date'].duplicated().any()
        if ambiguous:
            df.loc[group.index, 'burnout_order_uncertain'] = True
            continue
        group = group.sort_values(['_parsed_date', 'source_row'])
        pending = []
        run_number = 0

        def save_run():
            nonlocal run_number
            if len(pending) >= 2:
                run_number += 1
                df.loc[pending, 'burnout_detected'] = True
                df.loc[pending, 'burnout_streak'] = len(pending)
                df.loc[pending, 'burnout_sequence_id'] = f'{vol}|{case}|{run_number}'

        for ix, record in group.iterrows():
            flag_status = str(record.get('red_flags_status', ''))
            status_ok = (flag_status if flag_status not in ('', 'unprocessed') else str(record.get('analysis_status', ''))) == 'ok'
            marker = is_true(record.get('burnout_marker_detected', False))
            if status_ok and marker and str(record.get('text', '')).strip():
                pending.append(ix)
            else:
                save_run()
                pending = []
        save_run()
    return df.drop(columns=['_parsed_date'])