import asyncio
import subprocess
import sys
from typing import Tuple

import pandas as pd
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, TimeoutError


def ensure_chromium_installed() -> None:
    """
    Garante que o Chromium do Playwright esteja instalado.
    Se não estiver, instala via `playwright install chromium`.
    """
    try:
        subprocess.run(
            [sys.executable, "-m", "playwright", "install", "chromium"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print("✅ Playwright Chromium instalado.")
    except subprocess.CalledProcessError as e:
        print("❌ Falha ao instalar o Chromium:", e)
        sys.exit(1)


async def _scrape_year(page, year: int) -> pd.DataFrame:
    """
    Realiza a busca para um dado ano e retorna um DataFrame com os resultados,
    incluindo a linha de total.
    """
    # limpa e preenche o campo de pesquisa
    await page.fill("input.text_pesq", str(year), timeout=2000)
    await page.press("input.text_pesq", "Enter")
    # aguarda a tabela renderizar
    await page.wait_for_selector("table.tb_base.tb_dados", timeout=5000)

    # parse do HTML
    soup = BeautifulSoup(await page.content(), "lxml")
    table = soup.find("table", class_="tb_base tb_dados")
    if table is None:
        return pd.DataFrame()  # ou lançar erro, conforme sua necessidade

    # cabeçalhos
    headers = [th.get_text(strip=True) for th in table.thead.find_all("th")]

    # linhas do corpo
    rows = []
    for tr in table.tbody.find_all("tr"):
        cols = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(cols) == len(headers):
            rows.append(cols)

    # adiciona o total (tfoot)
    total_cells = [td.get_text(strip=True) for td in table.tfoot.find_all("td")]
    if len(total_cells) == len(headers):
        rows.append(total_cells)

    # monta DataFrame e marca o ano
    df = pd.DataFrame(rows, columns=headers)
    df["ano"] = year
    return df


async def fetch_production_data(interval: Tuple[int, int]) -> pd.DataFrame:
    """
    Para cada ano no intervalo [start, end], faz o scraping da aba 'Produção'
    e concatena todos os DataFrames num único.
    """
    start_year, end_year = sorted(interval)
    years = range(start_year, end_year + 1)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page()
        page.set_default_timeout(5000)
        page.set_default_navigation_timeout(5000)

        # navega até a página inicial
        await page.goto("http://vitibrasil.cnpuv.embrapa.br/")

        # clica em "Aceito" se aparecer o banner de cookies
        try:
            await page.click('button:has-text("Aceito")', timeout=2000)
        except TimeoutError:
            pass

        # seleciona a aba Produção
        await page.click('button:has-text("Produção")', timeout=2000)

        # coleta os dados ano a ano
        dfs = []
        for y in years:
            df_year = await _scrape_year(page, y)
            if not df_year.empty:
                dfs.append(df_year)

        await browser.close()

    # concatena e retorna
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def main():
    # 1) Garante que o Chromium está instalado
    ensure_chromium_installed()

    # 2) Define o intervalo de anos a buscar
    intervalo = (2015, 2020)

    # 3) Dispara o scraping e recebe um único DataFrame
    df = asyncio.run(fetch_production_data(intervalo))

    # 4) Exibe ou salva
    print(df)
    # df.to_csv("producoes_2015_2020.csv", index=False)


if __name__ == "__main__":
    main()
