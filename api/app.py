from contextlib import asynccontextmanager
from typing import List, Tuple, Sequence, Optional

import pandas as pd
from bs4 import BeautifulSoup
from fastapi import FastAPI, Query, HTTPException, Depends
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException
from starlette.concurrency import run_in_threadpool

from utils.auth import router as auth_router, get_current_active_user
from db.db_utils import init_db, save_scrape_results, query_scrape_results, ScrapeRecord

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Garante que o ChromeDriver esteja disponível no startup.
    """
    # Aqui você pode verificar se o chromedriver está no PATH, etc.
    yield


app = FastAPI(
    title="Embrapa Scraper API",
    description="Scraping com Selenium das abas do site da Embrapa com cache em SQLite e JWT",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(auth_router)


def _scrape_year_selenium(driver: webdriver.Chrome, year: int) -> pd.DataFrame:
    """
    Extrai os dados da tabela para um ano específico, usando Selenium.
    """
    # localiza e preenche o campo de pesquisa
    search_input = driver.find_element("css selector", "input.text_pesq")
    search_input.clear()
    search_input.send_keys(str(year))
    search_input.submit()  # ou driver.find_element(...).send_keys(Keys.ENTER)

    # aguarda a tabela aparecer
    try:
        driver.implicitly_wait(5)
        table = driver.find_element("css selector", "table.tb_base.tb_dados")
    except TimeoutException:
        return pd.DataFrame()

    html = table.get_attribute("outerHTML")
    soup = BeautifulSoup(html, "lxml")

    headers = [th.get_text(strip=True) for th in soup.select("thead th")]

    rows = []
    for tr in soup.select("tbody tr"):
        cols = [td.get_text(strip=True) for td in tr.select("td")]
        if len(cols) == len(headers):
            rows.append(cols)

    total = [td.get_text(strip=True) for td in soup.select("tfoot td")]
    if len(total) == len(headers):
        rows.append(total)

    df = pd.DataFrame(rows, columns=headers)
    df["ano"] = year
    return df


def fetch_scrape_selenium(
    interval: Tuple[int, int],
    tabs: Sequence[str]
) -> pd.DataFrame:
    """
    Executa o scraping via Selenium para cada ano no intervalo e concatena resultados.
    """
    start_year, end_year = sorted(interval)

    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(options=chrome_options)
    driver.set_window_size(1280, 800)
    driver.get("http://vitibrasil.cnpuv.embrapa.br/")

    wait = WebDriverWait(driver, 10)

    # clica em "Aceito" se aparecer o banner
    try:
        aceito_btn = wait.until(
            EC.element_to_be_clickable((By.XPATH, "//button[normalize-space(text())='Aceito']"))
        )
        aceito_btn.click()
    except:
        pass

    # navega pelas abas/subabas
    for label in tabs:
        try:
            btn = wait.until(
                EC.element_to_be_clickable((
                    By.XPATH,
                    f"//button[normalize-space(text())='{label}']"
                ))
            )
            btn.click()
        except:
            driver.quit()
            # Em vez de RuntimeError, disparar HTTPException pra FastAPI
            from fastapi import HTTPException
            raise HTTPException(
                status_code=404,
                detail=f"Aba/subaba '{label}' não encontrada"
            )

    dfs = []
    for year in range(start_year, end_year + 1):
        df_year = _scrape_year_selenium(driver, year)
        if not df_year.empty:
            dfs.append(df_year)

    driver.quit()
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def get_cached_years_db(
    session,
    main_tab: str,
    subtab: Optional[str]
) -> set[int]:
    """
    Retorna o conjunto de anos já persistidos no banco para a combinação de aba e subaba.
    """
    q = session.query(ScrapeRecord.year).filter(ScrapeRecord.tab == main_tab)
    if subtab:
        q = q.filter(ScrapeRecord.subtab == subtab)
    return {r[0] for r in q.distinct().all()}


@app.get(
    "/scrape",
    summary="Realiza scraping com cache e retorna JSON",
    dependencies=[Depends(get_current_active_user)]
)
async def scrape_endpoint(
    start: int = Query(..., ge=1970, le=2023, description="Ano inicial (até 2023)"),
    end:   int = Query(..., ge=1970, le=2023, description="Ano final (até 2023)"),
    tabs:  List[str] = Query(..., description="Caminho de abas/subabas, e.g. ['Produção','Vinhos de mesa']")
):
    """
    Retorna dados de scraping para as abas/subabas informadas,
    usando cache em SQLite para evitar buscas repetidas.
    """
    interval = (start, end)
    session = init_db("scraping.db")

    main_tab = tabs[0]
    subtab   = "/".join(tabs[1:]) if len(tabs) > 1 else None

    cached   = get_cached_years_db(session, main_tab, subtab)
    requested= set(range(start, end + 1))
    missing  = sorted(requested - cached)

    # para cada ano faltante, executa fetch_scrape_selenium em thread
    for year in missing:
        df_new = await run_in_threadpool(fetch_scrape_selenium, (year, year), tabs)
        if not df_new.empty:
            save_scrape_results(session, main_tab, df_new, subtab=subtab)

    df_all = query_scrape_results(session, main_tab, interval, subtab=subtab)
    return {"count": len(df_all), "data": df_all.to_dict(orient="records")}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
