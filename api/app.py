from fastapi import FastAPI, Query, HTTPException
from contextlib import asynccontextmanager
from typing import List, Tuple, Sequence
import subprocess
import sys

import pandas as pd
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, TimeoutError

from log_utils import load_query_log, log_query
from db.db_utils import init_db, save_scrape_results, query_scrape_results


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        subprocess.run(
            [sys.executable, "-m", "playwright", "install", "chromium"],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except subprocess.CalledProcessError as e:
        print("❌ Falha ao instalar o Chromium:", e)
        sys.exit(1)
    yield


app = FastAPI(
    title="Embrapa Scraper API",
    description="Scraping das abas do site da Embrapa com cache em SQLite",
    version="1.0.0",
    lifespan=lifespan,
)


async def _scrape_year(page, year: int) -> pd.DataFrame:
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
                raise HTTPException(status_code=404, detail=f"Aba/subaba '{label}' não encontrada")

        dfs = []
        for y in years:
            df_year = await _scrape_year(page, y)
            if not df_year.empty:
                dfs.append(df_year)

        await browser.close()

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def get_cached_years(tabs: Sequence[str]) -> set[int]:
    log = load_query_log()
    node = log
    for key in tabs:
        if isinstance(node, dict) and key in node:
            node = node[key]
        else:
            return set()
    return set(node) if isinstance(node, list) else set()


@app.get(
    "/scrape",
    summary="Realiza scraping com cache e retorna JSON"
)
async def scrape_endpoint(
    start: int = Query(..., ge=1970, le=2023, description="Ano inicial"),
    end: int = Query(..., ge=1970, le=2023, description="Ano final"),
    tabs: List[str] = Query(..., description="Caminho de abas e subabas, ex ['Produção','Vinhos de mesa']")
):
    interval = (start, end)
    session = init_db("scraping.db")

    cached = get_cached_years(tabs)
    requested = set(range(start, end + 1))
    missing = sorted(requested - cached)

    if missing:
        for year in missing:
            df_new = await fetch_scrape((year, year), tabs)
            if not df_new.empty:
                main_tab = tabs[0]
                subtab = "/".join(tabs[1:]) if len(tabs) > 1 else None
                save_scrape_results(session, main_tab, df_new, subtab=subtab)
            log_query(tabs, (year, year))

    main_tab = tabs[0]
    subtab = "/".join(tabs[1:]) if len(tabs) > 1 else None
    df_all = query_scrape_results(session, main_tab, interval, subtab=subtab)

    return {
        "count": len(df_all),
        "data": df_all.to_dict(orient="records")
    }
