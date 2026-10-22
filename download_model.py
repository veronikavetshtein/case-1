import shutil
import subprocess
from llm_client import LocalLLM

model = LocalLLM().model
if shutil.which('ollama') is None:
    raise SystemExit('Сначала установите Ollama с https://ollama.com/download')
print('Загрузка модели через официальный Ollama CLI:', model)
subprocess.run(['ollama', 'pull', model], check=True)
