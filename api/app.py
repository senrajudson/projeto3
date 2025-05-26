from fastapi import FastAPI, Query, HTTPException, Depends
from contextlib import asynccontextmanager
from typing import List, Tuple, Sequence, Optional
import subprocess
import sys

import pandas as pd
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, TimeoutError

from auth import router as auth_router, get_current_active_user
from db.db_utils import init_db, save_scrape_results, query_scrape_results, ScrapeRecord


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Garante que o Playwright Chromium esteja instalado no startup.
    """
    try:
        subprocess.run(
            [sys.executable, "-m", "playwright", "install", "chromium"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as e:
        print("❌ Falha ao instalar o Chromium:", e)
        sys.exit(1)
    yield


app = FastAPI(
    title="Embrapa Scraper API",
    description="Scraping das abas do site da Embrapa com cache em SQLite e JWT",
    version="1.0.0",
    lifespan=lifespan,
)

# registra o router de autenticação
app.include_router(auth_router)


async def _scrape_year(page, year: int) -> pd.DataFrame:
    """
    Extrai os dados da tabela para um ano específico.
    """
    await page.fill("input.text_pesq", str(year), timeout=2000)
    await page.press("input.text_pesq", "Enter")

    try:
        await page.wait_for_selector("table.tb_base.tb_dados", timeout=5000)
    except TimeoutError:
        return pd.DataFrame()

    soup = BeautifulSoup(await page.content(), "lxml")
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


async def fetch_scrape(
    interval: Tuple[int, int],
    tabs: Sequence[str]
) -> pd.DataFrame:
    """
    Executa o scraping para cada ano no intervalo e concatena os resultados.
    """
    start_year, end_year = sorted(interval)
    years = range(start_year, end_year + 1)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page()
        page.set_default_timeout(5000)
        page.set_default_navigation_timeout(5000)

        await page.goto("http://vitibrasil.cnpuv.embrapa.br/", wait_until="networkidle")

        try:
            await page.click('button:has-text("Aceito")', timeout=2000)
        except TimeoutError:
            pass

        for label in tabs:
            try:
                await page.click(f'button:has-text("{label}")', timeout=2000)
            except TimeoutError:
                raise HTTPException(status_code=404,
                                    detail=f"Aba/subaba '{label}' não encontrada")

        dfs = []
        for year in years:
            df_year = await _scrape_year(page, year)
            if not df_year.empty:
                dfs.append(df_year)

        await browser.close()

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def get_cached_years_db(
    session,
    main_tab: str,
    subtab: Optional[str]
) -> set[int]:
    """
    Retorna o conjunto de anos já persistidos no banco para a combinação
    de aba e subaba informadas.
    """
    query = session.query(ScrapeRecord.year).filter(
        ScrapeRecord.tab == main_tab
    )
    if subtab:
        query = query.filter(ScrapeRecord.subtab == subtab)
    results = query.distinct().all()
    return {r[0] for r in results}


@app.get(
    "/scrape",
    summary="Realiza scraping com cache e retorna JSON",
    dependencies=[Depends(get_current_active_user)]
)
async def scrape_endpoint(
    start: int = Query(..., ge=1970, le=2023, description="Ano inicial (até 2023)"),
    end: int = Query(..., ge=1970, le=2023, description="Ano final (até 2023)"),
    tabs: List[str] = Query(
        ..., description="Caminho de abas e subabas, ex: ['Produção','Vinhos de mesa']"
    )
):
    """
    Retorna dados de scraping para as abas/subabas informadas,
    usando cache em SQLite para evitar buscas repetidas.
    """
    interval = (start, end)
    session = init_db("scraping.db")

    main_tab = tabs[0]
    subtab = "/".join(tabs[1:]) if len(tabs) > 1 else None

    cached = get_cached_years_db(session, main_tab, subtab)
    requested = set(range(start, end + 1))
    missing = sorted(requested - cached)

    if missing:
        for year in missing:
            df_new = await fetch_scrape((year, year), tabs)
            if not df_new.empty:
                save_scrape_results(session, main_tab, df_new, subtab=subtab)

    df_all = query_scrape_results(session, main_tab, interval, subtab=subtab)

    return {
        "count": len(df_all),
        "data": df_all.to_dict(orient="records")
    }

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.app:app",               # módulo e app FastAPI
        host="10.247.179.197",           # escuta em todas as interfaces
        port=8000,                # porta de serviço
        reload=True,              # reinício automático em código alterado
        log_level="info"          # nível de logs
    )