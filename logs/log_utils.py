import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

# Arquivo de histórico de consultas
LOG_FILE = Path("query_log.json")


def load_query_log() -> Dict[str, Any]:
    """
    Carrega o histórico de consultas do arquivo JSON.
    Retorna um dicionário vazio se o arquivo não existir ou estiver corrompido.
    """
    if LOG_FILE.exists():
        try:
            return json.loads(LOG_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def write_query_log(log: Dict[str, Any]) -> None:
    """
    Escreve o histórico de consultas no arquivo JSON.
    """
    LOG_FILE.write_text(
        json.dumps(log, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


def log_query(
    tab: Union[str, List[str], Tuple[str, ...]],
    interval: Tuple[int, int]
) -> None:
    """
    Registra no JSON que fizemos consultas para a aba ou caminho de abas `tab`
    para cada ano do intervalo [start, end], sem duplicar entradas.

    `tab` pode ser uma string única (ex.: "Produção") ou uma lista/tupla
    para representar abas e subabas (ex.: ["Produção", "Vinhos de mesa"]).
    """
    start, end = sorted(interval)
    log = load_query_log()

    # Normaliza tab em sequência de chaves
    seq = list(tab) if isinstance(tab, (list, tuple)) else [tab]

    # Navega pelo dicionário aninhado conforme as chaves em seq
    pointer: Dict[str, Any] = log
    for key in seq[:-1]:
        if key not in pointer or not isinstance(pointer[key], dict):
            pointer[key] = {}
        pointer = pointer[key]  # tipo: ignore

    leaf = seq[-1]
    if leaf not in pointer or not isinstance(pointer[leaf], list):
        pointer[leaf] = []
    years_list: List[int] = pointer[leaf]  # tipo: ignore

    # Adiciona apenas anos não existentes
    existing_years = set(years_list)
    new_years: List[int] = []
    for year in range(start, end + 1):
        if year not in existing_years:
            years_list.append(year)
            new_years.append(year)

    if new_years:
        # Mantém a lista ordenada
        years_list.sort()
        write_query_log(log)
        path_str = " > ".join(seq)
        print(f"✅ Registrados novos anos para '{path_str}': {new_years}")
    else:
        path_str = " > ".join(seq)
        print(f"ℹ️ Todos os anos de {start} a {end} já estavam registrados para '{path_str}'.")
