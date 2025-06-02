from contextlib import asynccontextmanager
from typing import List, Tuple, Sequence, Optional
import requests, json
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from fastapi import FastAPI, Query, Depends, HTTPException
from starlette.concurrency import run_in_threadpool

from utils.auth import router as auth_router, get_current_active_user
from db.db_utils import init_db, save_scrape_results, query_scrape_results, ScrapeRecord

BASE_DIR = Path(__file__).parent
json_path = BASE_DIR / "references.json"

references_dict = []
with open(json_path, "r", encoding="utf-8") as f:
    references_dict = json.load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="Embrapa Scraper API",
    description="Scraping com requests+BS4 das abas do site da Embrapa com cache em SQLite e JWT",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(auth_router)


def _scrape_year_bs4(base_url: str, year: int) -> pd.DataFrame:
    try:
        response = requests.post(base_url, data={"ano": str(year)}, timeout=120)
        response.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Erro ao acessar Embrapa: {e}")

    soup = BeautifulSoup(response.text, "lxml")

    table = soup.select_one("table.tb_base.tb_dados")
    if not table:
        return pd.DataFrame()

    headers = [th.get_text(strip=True) for th in table.select("thead th")]

    rows = []
    for tr in table.select("tbody tr"):
        cols = [td.get_text(strip=True) for td in tr.select("td")]
        if len(cols) == len(headers):
            rows.append(cols)

    total = [td.get_text(strip=True) for td in table.select("tfoot td")]
    if len(total) == len(headers):
        rows.append(total)

    df = pd.DataFrame(rows, columns=headers)
    df["ano"] = year
    return df


def resolve_url_from_tabs(tabs: Sequence[str]) -> Optional[str]:
    node = references_dict
    for key in tabs:
        if isinstance(node, dict) and key in node:
            node = node[key]
        else:
            return None
    return node if isinstance(node, str) else None


def fetch_scrape_bs4(interval: Tuple[int, int], tabs: Sequence[str]) -> pd.DataFrame:
    target_url = resolve_url_from_tabs(tabs)
    if target_url is None:
        raise HTTPException(
            status_code=404, detail=f"Aba/subaba {tabs!r} não encontrada no mapeamento"
        )

    start_year, end_year = sorted(interval)
    dfs = []
    for year in range(start_year, end_year + 1):
        df_year = _scrape_year_bs4(target_url, year)
        if not df_year.empty:
            dfs.append(df_year)

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def get_cached_years_db(session, main_tab: str, subtab: Optional[str]) -> set[int]:
    q = session.query(ScrapeRecord.year).filter(ScrapeRecord.tab == main_tab)
    if subtab:
        q = q.filter(ScrapeRecord.subtab == subtab)
    return {r[0] for r in q.distinct().all()}


@app.get(
    "/scrape",
    summary="Realiza scraping com cache e retorna JSON",
    dependencies=[Depends(get_current_active_user)],
)
async def scrape_endpoint(
    start: int = Query(..., ge=1970, le=2023, description="Ano inicial (até 2023)"),
    end: int = Query(..., ge=1970, le=2023, description="Ano final (até 2023)"),
    tabs: List[str] = Query(
        ..., description="Caminho de abas/subabas, e.g. ['Produção','Vinhos de mesa']"
    ),
):
    interval = (start, end)
    session = init_db("scraping.db")

    main_tab = tabs[0]
    subtab = "/".join(tabs[1:]) if len(tabs) > 1 else None

    cached = get_cached_years_db(session, main_tab, subtab)
    requested = set(range(start, end + 1))
    missing = sorted(requested - cached)

    for year in missing:
        df_new = await run_in_threadpool(fetch_scrape_bs4, (year, year), tabs)
        if not df_new.empty:
            save_scrape_results(session, main_tab, df_new, subtab=subtab)

    df_all = query_scrape_results(session, main_tab, interval, subtab=subtab)
    return {"count": len(df_all), "data": df_all.to_dict(orient="records")}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
