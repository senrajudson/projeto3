import asyncio
import subprocess
import sys
from typing import Tuple, Sequence

import pandas as pd
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, TimeoutError

from log_utils import log_query  # importamos apenas o que precisamos

def ensure_chromium_installed() -> None:
    """
    Garante que o Chromium do Playwright esteja instalado.
    Se não estiver, instala via `playwright install chromium`.
    """
    try:
        subprocess.run(
            [sys.executable, "-m", "playwright", "install", "chromium"],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as e:
        print("❌ Falha ao instalar o Chromium:", e)
        sys.exit(1)


async def _scrape_year(page, year: int) -> pd.DataFrame:
    """
    Busca dados de um ano, monta DataFrame com as linhas de corpo + total.
    """
    await page.fill("input.text_pesq", str(year), timeout=2000)
    await page.press("input.text_pesq", "Enter")
    try:
        await page.wait_for_selector("table.tb_base.tb_dados", timeout=5000)
    except TimeoutError:
        print(f"Ano {year}: tabela não carregou a tempo.")
        return pd.DataFrame()

    soup = BeautifulSoup(await page.content(), "lxml")
    table = soup.select_one("table.tb_base.tb_dados")
    if not table:
        return pd.DataFrame()

    headers = [th.get_text(strip=True) for th in table.select("thead th")]
    rows = [
        [td.get_text(strip=True) for td in tr.select("td")]
        for tr in table.select("tbody tr")
    ]
    rows = [r for r in rows if len(r) == len(headers)]

    total = [td.get_text(strip=True) for td in table.select("tfoot td")]
    if len(total) == len(headers):
        rows.append(total)

    df = pd.DataFrame(rows, columns=headers)
    df["ano"] = year
    return df


async def fetch_production_data(
    interval: Tuple[int, int],
    info: Sequence[str] = ("Produção",)
) -> pd.DataFrame:
    """
    Faz scraping da(s) aba(s) em `info` para todos os anos do intervalo,
    concatena e retorna um DataFrame único.
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

        for tab_label in info:
            try:
                await page.click(f'button:has-text("{tab_label}")', timeout=2000)
            except TimeoutError:
                print(f"Aba '{tab_label}' não encontrada.")

        dfs = []
        for year in years:
            df_year = await _scrape_year(page, year)
            if not df_year.empty:
                dfs.append(df_year)

        await browser.close()

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def main() -> None:
    ensure_chromium_installed()

    intervalo = (2015, 2020)
    aba = "Produção"

    df = asyncio.run(fetch_production_data(intervalo, info=(aba,)))
    log_query(aba, intervalo)

    print(df)
    # df.to_csv("producoes_2015_2020.csv", index=False)


if __name__ == "__main__":
    main()
