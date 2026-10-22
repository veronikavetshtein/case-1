"""Editable criterion names/descriptions, keeping the core 16 keys stable."""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path
from criteria import ALL_CRITERIA, CRITERIA_GROUPS

HERE = Path(__file__).resolve().parent
STORE = HERE / 'criteria_config.json'


def defaults():
    return [dict(key=k, title=v, description=v, enabled=True, core=True)
            for k, v in ALL_CRITERIA.items()]


def load_config():
    base = {v['key']: v for v in defaults()}
    if STORE.exists():
        loaded = json.loads(STORE.read_text(encoding='utf-8'))
        if not isinstance(loaded, list):
            raise ValueError('criteria_config.json должен содержать список критериев')
        for item in loaded:
            key = item.get('key', '')
            if not re.fullmatch(r'[a-z][a-z0-9_]{1,40}', key):
                raise ValueError(f'Некорректный идентификатор: {key}')
            title = str(item.get('title', '')).strip()[:200]
            description = str(item.get('description', '')).strip()[:1500]
            if not title or not description:
                raise ValueError(f'Не заполнено описание критерия: {key}')
            if key in base:
                base[key]['title'] = title
                base[key]['description'] = description
                base[key]['enabled'] = True  # cannot silently disable mandatory core
            else:
                if not key.startswith('custom_'):
                    raise ValueError('Дополнительный критерий должен начинаться на custom_')
                base[key] = dict(key=key, title=title, description=description,
                                 enabled=bool(item.get('enabled', True)), core=False)
    return list(base.values())


def save_config(items):
    core_keys = set(ALL_CRITERIA)
    seen = set()
    for item in items:
        key = item.get('key')
        if key in seen:
            raise ValueError('Дублирующиеся ключи')
        seen.add(key)
        if key in core_keys and not item.get('enabled', True):
            raise ValueError('Нельзя отключать обязательные 16 критериев')
        if key not in core_keys and not str(key).startswith('custom_'):
            raise ValueError('Ключ пользовательского критерия должен начинаться с custom_')
    if not core_keys.issubset(seen):
        raise ValueError('Отсутствуют обязательные критерии')
    STORE.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding='utf-8')


def active_criteria():
    return {x['key']: x['description'] for x in load_config() if x['enabled']}


def criteria_groups():
    all_active = active_criteria()
    groups = [(name, {k: all_active[k] for k in criteria if k in all_active})
              for name, criteria in CRITERIA_GROUPS]
    custom = {k: v for k, v in all_active.items() if k not in ALL_CRITERIA}
    if custom:
        groups.append(('Дополнительные критерии', custom))
    return groups


def config_fingerprint(model, policy_text=''):
    payload = json.dumps({'criteria': active_criteria(), 'model': model,
                          'policy': policy_text.strip()}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]
