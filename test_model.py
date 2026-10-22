"""Проверка локальной Ollama на одном отчёте."""
import argparse
from parser import parse_excel
from llm_client import LocalLLM

p = argparse.ArgumentParser()
p.add_argument('--excel', default='reports.xlsx')
args = p.parse_args()

llm = LocalLLM()
llm.ensure_model()
df = parse_excel(args.excel)
candidates = df.loc[df.text.str.len() > 100]
if candidates.empty:
    raise SystemExit('Нет подходящих отчетов длиной >100 символов.')
report = candidates.iloc[0]
print('Исходная строка:', report.source_row)
print('Текст:', report.text[:300])
result = llm.analyze_report(report.text)
print('Анализ:', result)
