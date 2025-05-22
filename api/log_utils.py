import json
from pathlib import Path
from typing import Tuple

# Arquivo de histórico de consultas
LOG_FILE = Path("query_log.json")


def load_query_log() -> list:
    """
    Carrega o histórico de consultas do arquivo JSON.
    Retorna lista vazia se o arquivo não existir ou estiver corrompido.
    """
    if LOG_FILE.exists():
        try:
            return json.loads(LOG_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
    return []


def write_query_log(log: list) -> None:
    """
    Escreve o histórico de consultas no arquivo JSON.
    """
    LOG_FILE.write_text(
        json.dumps(log, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


def log_query(tab: str, interval: Tuple[int, int]) -> None:
    """
    Registra no JSON que fizemos uma consulta para a aba `tab`
    para cada ano do intervalo [start, end], sem duplicar entradas.
    """
    start, end = sorted(interval)
    log = load_query_log()
    # Identifica anos já registrados para esta aba
    existing_years = {
        entry["year"]
        for entry in log
        if entry.get("tab") == tab and "year" in entry
    }
    new_years = []
    # Registra individualmente cada ano
    for year in range(start, end + 1):
        if year not in existing_years:
            entry = {"tab": tab, "year": year}
            log.append(entry)
            new_years.append(year)

    if new_years:
        write_query_log(log)
        print(f"✅ Anos registrados para aba '{tab}': {new_years}")
    else:
        print(
            f"ℹ️ Todos os anos de {start} a {end} para aba '{tab}' já estavam registrados."
        )
