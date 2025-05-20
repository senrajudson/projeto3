from playwright.sync_api import sync_playwright, Error
import subprocess
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


if __name__ == "__main__":
   
   if is_chromium_installed():
    print("Is installed!")
   else:
    subprocess.run([sys.executable, "-m", 'playwright', 'install', 'chromium'], check=True)
    print("Now it's installed!")