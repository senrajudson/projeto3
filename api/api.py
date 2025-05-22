import asyncio
import subprocess
import sys
from typing import Sequence, Tuple

import pandas as pd
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, TimeoutError


def ensure_chromium_installed() -> None:
    """
    Garante que o Playwright Chromium está disponível.
    Se não estiver, instala via `playwright install chromium`.
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


async def _scrape_year(page, year: int, *,
                       table_timeout: int = 5_000) -> pd.DataFrame:
    """
    Preenche o campo de ano, aguarda a tabela e retorna um DataFrame
    com as linhas de <tbody> + a linha de total de <tfoot>.
    """
    # preenche e submete
    await page.fill("input.text_pesq", str(year), timeout=2_000)
    await page.press("input.text_pesq", "Enter")

    # espera a tabela aparecer
    try:
        await page.wait_for_selector("table.tb_base.tb_dados", timeout=table_timeout)
    except TimeoutError:
        print(f"Ano {year}: tabela não carregou em {table_timeout} ms")
        return pd.DataFrame()

    html = await page.content()
    soup = BeautifulSoup(html, "lxml")
    table = soup.select_one("table.tb_base.tb_dados")
    if not table:
        return pd.DataFrame()

    # cabeçalho
    headers = [th.get_text(strip=True) for th in table.select("thead th")]

    # corpo
    data = []
    for row in table.select("tbody tr"):
        cols = [td.get_text(strip=True) for td in row.select("td")]
        if len(cols) == len(headers):
            data.append(cols)

    # total
    total = [td.get_text(strip=True) for td in table.select("tfoot td")]
    if len(total) == len(headers):
        data.append(total)

    df = pd.DataFrame(data, columns=headers)
    df["ano"] = year
    return df


async def fetch_tab_data(
    base_url: str,
    years: Sequence[int],
    *,
    cookie_button_text: str = "Aceito",
    tab_label: str = "Produção",
) -> pd.DataFrame:
    """
    Abre o browser, navega até base_url, aceita cookies (se houver),
    clica na aba indicada por tab_label e faz _scrape_year para cada ano.
    Retorna o concat de todos os DataFrames válidos.
    """
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page()
        page.set_default_timeout(5_000)
        page.set_default_navigation_timeout(5_000)

        await page.goto(base_url)

        # aceita cookies sem travar no erro
        try:
            await page.click(f'button:has-text("{cookie_button_text}")', timeout=2_000)
        except TimeoutError:
            pass

        # seleciona a aba
        await page.click(f'button:has-text("{tab_label}")', timeout=2_000)

        # percorre os anos
        dfs = []
        for y in years:
            df = await _scrape_year(page, y)
            if not df.empty:
                dfs.append(df)

        await browser.close()

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def main(interval: Tuple[int, int]) -> None:
    # garante o browser instalado
    ensure_chromium_installed()

    # monta lista de anos
    start, end = sorted(interval)
    years = list(range(start, end + 1))

    # URL base e aba desejada
    url = "http://vitibrasil.cnpuv.embrapa.br/"
    df = asyncio.run(fetch_tab_data(url, years, tab_label="Produção"))

    # saída
    print(df)
    # df.to_csv("producoes.csv", index=False)


if __name__ == "__main__":
    # ex: rodar de 2015 até 2020
    main((2015, 2020))
