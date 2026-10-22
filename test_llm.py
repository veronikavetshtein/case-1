from llm_client import LocalLLM
from parser import parse_excel

print("Загружаем данные...")
df = parse_excel("reports.xlsx")
# Берем первый попавшийся отчет с длинным текстом
sample_text = df[df['text'].str.len() > 100].iloc[0]['text']

print(f"\n--- ТЕСТОВЫЙ ТЕКСТ ---\n{sample_text[:300]}...\n")

llm = LocalLLM()
print("Отправляем запрос к локальной модели (ожидание 10-30 сек)...")
result = llm.analyze_report(sample_text)

print("\n--- РЕЗУЛЬТАТ РАЗМЕТКИ ---")
print(result)