from fastapi import FastAPI, Query, HTTPException
from contextlib import asynccontextmanager
from typing import List, Tuple, Sequence
# import asyncio
import subprocess
import sys

import pandas as pd
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, TimeoutError

from log_utils import log_query


# Lifespan handler para startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: garante instalação do Chromium
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
    # Shutdown: nada a fazer


app = FastAPI(
    title="Embrapa Scraper API",
    description="API para realizar scraping das abas do site da Embrapa e logar consultas",
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
                print(f"Aba/subaba '{label}' não encontrada.")

        dfs = []
        for y in years:
            df_year = await _scrape_year(page, y)
            if not df_year.empty:
                dfs.append(df_year)

        await browser.close()

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


@app.get("/scrape", summary="Realiza scraping e retorna dados em JSON")
async def scrape_endpoint(
    start: int = Query(..., ge=1970, le=2023, description="Ano inicial da consulta"),
    end: int = Query(..., ge=1970, le=2023, description="Ano final da consulta"),
    tabs: List[str] = Query(
        ..., description="Lista de abas/subabas (ex: ['Produção','Vinhos de mesa'])"
    ),
):
    interval = (start, end)
    try:
        df = await fetch_scrape(interval, tabs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no scraping: {e}")

    for tab_path in tabs:
        log_query(tab_path, interval)

    data = df.to_dict(orient="records")
    return {"count": len(data), "data": data}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_api:app", host="0.0.0.0", port=8000, reload=True)
