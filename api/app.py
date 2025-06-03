# api/app.py

from contextlib import asynccontextmanager
from typing import List, Tuple, Sequence, Optional
import os
import requests
import json
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from fastapi import FastAPI, Query, Depends, HTTPException
from starlette.concurrency import run_in_threadpool

from mangum import Mangum

from utils.auth import router as auth_router, get_current_active_user

# Importa as funções de banco mas só será usado se NÃO estivermos no Vercel
from db.db_utils import init_db, save_scrape_results, query_scrape_results, ScrapeRecord

# Carrega references.json (mapeamento de abas/subabas → URLs)
BASE_DIR = Path(__file__).parent
json_path = BASE_DIR / "references.json"
with open(json_path, "r", encoding="utf-8") as f:
    references_dict = json.load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan handler: sem ações específicas no startup/shutdown
    (Mangum/Lambda irá gerenciar o ciclo de vida).
    """
    yield


app = FastAPI(
    title="Embrapa Scraper API",
    description=(
        "Scraping com requests+BeautifulSoup das abas do site da Embrapa "
        "com opção de cache em SQLite e JWT (rodando em Vercel + Mangum)"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Registra as rotas de autenticação (POST /users, POST /token)
app.include_router(auth_router)


def _scrape_year_bs4(base_url: str, year: int) -> pd.DataFrame:
    """
    Extrai a tabela do ano específico a partir de uma URL base (requests + BS4).
    Lança HTTPException(503) se der erro de rede ou timeout.
    """
    try:
        response = requests.post(base_url, data={"ano": str(year)}, timeout=120)
        response.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Erro ao acessar Embrapa: {e}")

    soup = BeautifulSoup(response.text, "lxml")
    table = soup.select_one("table.tb_base.tb_dados")
    if not table:
        return pd.DataFrame()

    # Extrai cabeçalhos (thead > th)
    headers = [th.get_text(strip=True) for th in table.select("thead th")]

    # Extrai linhas do corpo (tbody > tr > td)
    rows: List[List[str]] = []
    for tr in table.select("tbody tr"):
        cols = [td.get_text(strip=True) for td in tr.select("td")]
        if len(cols) == len(headers):
            rows.append(cols)

    # Extrai total (tfoot > td), se existir
    total = [td.get_text(strip=True) for td in table.select("tfoot td")]
    if len(total) == len(headers):
        rows.append(total)

    df = pd.DataFrame(rows, columns=headers)
    df["ano"] = year
    return df


def resolve_url_from_tabs(tabs: Sequence[str]) -> Optional[str]:
    """
    Dado um caminho de abas/subabas (lista de strings), faz lookup em references_dict
    e devolve a URL correspondente (string), ou None se não encontrar.
    """
    node = references_dict
    for key in tabs:
        if isinstance(node, dict) and key in node:
            node = node[key]
        else:
            return None
    return node if isinstance(node, str) else None


def fetch_scrape_bs4(interval: Tuple[int, int], tabs: Sequence[str]) -> pd.DataFrame:
    """
    Executa _scrape_year_bs4 ano a ano no intervalo e concatena num DataFrame.
    Se não encontrar URL para estas tabs, lança 404.
    """
    target_url = resolve_url_from_tabs(tabs)
    if target_url is None:
        raise HTTPException(
            status_code=404,
            detail=f"Aba/subaba {tabs!r} não encontrada no mapeamento"
        )

    start_year, end_year = sorted(interval)
    dfs: List[pd.DataFrame] = []
    for year in range(start_year, end_year + 1):
        df_year = _scrape_year_bs4(target_url, year)
        if not df_year.empty:
            dfs.append(df_year)

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def get_cached_years_db(session, main_tab: str, subtab: Optional[str]) -> set[int]:
    """
    Retorna o conjunto de anos já persistidos no banco para a combinação de aba e subaba.
    """
    query = session.query(ScrapeRecord.year).filter(ScrapeRecord.tab == main_tab)
    if subtab:
        query = query.filter(ScrapeRecord.subtab == subtab)
    results = query.distinct().all()
    return {r[0] for r in results}


@app.get(
    "/scrape",
    summary="Realiza scraping com cache (se habilitado) e retorna JSON",
    dependencies=[Depends(get_current_active_user)],
)
async def scrape_endpoint(
    start: int = Query(..., ge=1970, le=2023, description="Ano inicial (até 2023)"),
    end:   int = Query(..., ge=1970, le=2023, description="Ano final (até 2023)"),
    tabs:  List[str] = Query(
        ..., description="Caminho de abas/subabas, ex: ['Produção','Vinhos de mesa']"
    )
):
    """
    Retorna dados de scraping para as abas/subabas informadas.

    **Funcionalidades**:
    - Verifica disponibilidade do site da Embrapa (503 se indisponível).
    - Se NÃO estivermos rodando em Vercel (variável VERCEL != "1"), usa SQLite para cache:
        1. Checa quais anos deste intervalo JÁ estão no banco.
        2. Para cada ano faltante, faz scraping (requests+BS4) em background thread.
        3. Salva resultados novos no SQLite.
        4. Retorna todos os anos do intervalo vindos do banco.
    - Se estivermos em Vercel (VERCEL="1"), PULA o cache em SQLite:
        1. Simplesmente faz scraping de todos os anos (requests+BS4) na hora.
        2. Não grava/consulta nada no SQLite.
    """
    # 1) Checa disponibilidade do site (fallback rápido)
    try:
        requests.get("http://vitibrasil.cnpuv.embrapa.br/", timeout=15).raise_for_status()
    except Exception:
        raise HTTPException(status_code=503, detail="Site da Embrapa indisponível")

    interval = (start, end)
    running_on_vercel = os.environ.get("VERCEL") == "1"

    # Se NÃO estivermos em Vercel, inicializa o banco
    if not running_on_vercel:
        session = init_db("scraping.db")
    else:
        session = None  # não usaremos

    main_tab = tabs[0]
    subtab = "/".join(tabs[1:]) if len(tabs) > 1 else None

    # **Lógica de cache só se session não for None**:
    if session is not None:
        # 2a) Determina anos já cacheados
        cached = get_cached_years_db(session, main_tab, subtab)
        requested = set(range(start, end + 1))
        missing = sorted(requested - cached)

        # 3a) Para cada ano faltante, faz scraping e salva no banco
        for year in missing:
            df_new = await run_in_threadpool(fetch_scrape_bs4, (year, year), tabs)
            if not df_new.empty:
                save_scrape_results(session, main_tab, df_new, subtab=subtab)

        # 4a) Recupera do banco todos os anos no intervalo
        df_all = query_scrape_results(session, main_tab, interval, subtab=subtab)

    else:
        # **Estamos em Vercel**: pulamos o cache e fazemos scraping direto de todos os anos
        df_list: List[pd.DataFrame] = []
        start_year, end_year = sorted(interval)
        for year in range(start_year, end_year + 1):
            # cada ano separado
            df_year = await run_in_threadpool(fetch_scrape_bs4, (year, year), tabs)
            if not df_year.empty:
                df_list.append(df_year)
        if df_list:
            df_all = pd.concat(df_list, ignore_index=True)
        else:
            df_all = pd.DataFrame()

    return {
        "count": len(df_all),
        "data": df_all.to_dict(orient="records")
    }


# Handler WSGI para AWS Lambda / Vercel
handler = Mangum(app)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
