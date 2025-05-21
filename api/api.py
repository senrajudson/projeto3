from playwright.async_api import async_playwright, TimeoutError
from playwright.sync_api import sync_playwright, Error
import subprocess
import asyncio
import sys

  
def is_chromium_installed() -> bool:
    try:
      with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        browser.close()
      return True
    except Error as e:
      if "install" in e.message.lower():
         return False
      

async def producoes_resultados(periodo:list, tipo:str = 'all'):
    async with async_playwright() as p:
        browser = await p.chromium.launch(channel='chrome', headless=True)
        page = await browser.new_page()

        page.set_default_timeout(5_000)
        page.set_default_navigation_timeout(5_000)

        await page.goto('http://vitibrasil.cnpuv.embrapa.br/')

        try:
            await page.click('button:hast-text("Aceito")', timeout=2_000)
            pass
        except TimeoutError:
            pass
        
        # # 3) Preenche a barra de pesquisa e envia
        # await page.fill('input[name="q"]', query)
        # await page.press('input[name="q"]', "Enter")

        await page.click('button:hast-text("Produção")', timeout=2_000)


        async def scrap_results(periodo:list, i:int):
            await page.click('input[class~="text_pesq"]', timeout=1_000)

            await page.fill('input[class~="text_pesq"]', periodo[i])

            await page.press('input[class~="text_pesq"]', "Enter")


        

        # 4) Aguarda a área de resultados
        await page.wait_for_selector("#search")

        # 5) Coleta os 10 primeiros resultados
        results = await page.query_selector_all('#search .g')
        for idx, result in enumerate(results[:10], start=1):
            title_el = await result.query_selector('h3')
            title = await title_el.inner_text() if title_el else "—"
            link_el = await result.query_selector('a')
            href = await link_el.get_attribute('href') if link_el else "—"
            print(f"{idx}. {title}\n   {href}\n")

        await browser.close()


async def search_google(query: str):
    async with async_playwright() as p:
        # Lança o Chrome instalado no sistema em modo headless
        browser = await p.chromium.launch(channel="chrome", headless=True)
        page = await browser.new_page()

        # Define timeouts padrão (em ms)
        page.set_default_timeout(20_000)            # 20 s para ações genéricas
        page.set_default_navigation_timeout(30_000) # 30 s para navegações

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
        results = await page.query_selector_all('#search .g')
        for idx, result in enumerate(results[:10], start=1):
            title_el = await result.query_selector('h3')
            title = await title_el.inner_text() if title_el else "—"
            link_el = await result.query_selector('a')
            href = await link_el.get_attribute('href') if link_el else "—"
            print(f"{idx}. {title}\n   {href}\n")

        await browser.close()


if __name__ == "__main__":
   
  if is_chromium_installed():
    print("Is installed!")
  else:
    subprocess.run([sys.executable, "-m", 'playwright', 'install', 'chromium'], check=True)
    print("Now it's installed!")

  termo = "playwright python scraping"
  asyncio.run(search_google(termo))