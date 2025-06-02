async def search_google(query: str):
    async with async_playwright() as p:
        # Lança o Chrome instalado no sistema em modo headless
        browser = await p.chromium.launch(channel="chrome", headless=True)
        page = await browser.new_page()

        # Define timeouts padrão (em ms)
        page.set_default_timeout(20_000)  # 20 s para ações genéricas
        page.set_default_navigation_timeout(30_000)  # 30 s para navegações

        # 1) Navega até o Google
        await page.goto("https://www.google.com")
        # 2) Fecha banner de cookies, se existir
        try:
            await page.click('button:has-text("Aceito")', timeout=2_000)
        except TimeoutError:
            pass

        # 3) Preenche a barra de pesquisa e envia
        await page.fill('input[name="q"]', query)
        await page.press('input[name="q"]', "Enter")

        # 4) Aguarda a área de resultados
        await page.wait_for_selector("#search")

        # 5) Coleta os 10 primeiros resultados
        results = await page.query_selector_all("#search .g")
        for idx, result in enumerate(results[:10], start=1):
            title_el = await result.query_selector("h3")
            title = await title_el.inner_text() if title_el else "—"
            link_el = await result.query_selector("a")
            href = await link_el.get_attribute("href") if link_el else "—"
            print(f"{idx}. {title}\n   {href}\n")

        await browser.close()


async def scrape_tabela(url: str) -> pd.DataFrame:
    # 1) Busca o HTML da página via Playwright
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url)
        html = await page.content()
        await browser.close()

    # 2) Parseia com BeautifulSoup
    soup = BeautifulSoup(html, "lxml")
    tabela = soup.find("table", class_="tb_base tb_dados")

    # 3) Extrai cabeçalho (opcional)
    headers = [th.get_text(strip=True) for th in tabela.thead.find_all("th")]

    # 4) Extrai linhas do corpo
    rows = []
    for tr in tabela.tbody.find_all("tr"):
        cols = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(cols) == len(headers):
            rows.append(dict(zip(headers, cols)))

    # 5) Cria DataFrame
    df = pd.DataFrame(rows, columns=headers)
    return df
