# pip install selenium==4.14.0
# pip install webdriver-manager==4.0.1
import json
from typing import List, Dict

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from webdriver_manager.firefox import GeckoDriverManager


def save_json(path: str, data: List[dict]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def parse_table(driver: webdriver.Firefox, table_xpath: str) -> List[Dict]:
    """Парсит таблицу вида <table><tbody><tr><td>... без заголовков."""
    table = driver.find_element(By.XPATH, table_xpath)
    rows = table.find_elements(By.XPATH, ".//tbody/tr[td]") or table.find_elements(By.XPATH, ".//tr[td]")
    parsed: List[Dict] = []

    for r in rows:
        cells = r.find_elements(By.XPATH, "./td") or r.find_elements(By.XPATH, "./th")
        if not cells:
            continue

        row: Dict[str, str] = {}
        for i, cell in enumerate(cells, start=1):
            key = f"col_{i}"
            text = cell.text.strip()
            links = cell.find_elements(By.XPATH, ".//a[@href]")
            if links:
                row[key] = text
                row[f"{key}_link"] = links[0].get_attribute("href")
            else:
                row[key] = text

        if any(v for v in row.values()):
            parsed.append(row)

    return parsed


if __name__ == "__main__":
    URL = "https://oliis.jinr.ru/index.php/patentovanie-2/8-russian/25-dejstvuyushchie-patenty-oiyai"

    service = FirefoxService(executable_path=GeckoDriverManager().install())
    driver = webdriver.Firefox(service=service)
    driver.set_window_size(1280, 900)

    TABLE_XPATH = "(//table)[2]"  # <-- ВАЖНО: скобки + индекс 2 = вторая таблица

    try:
        driver.get(URL)

        # ждём именно вторую таблицу и хотя бы одну строку с <td>
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.XPATH, TABLE_XPATH))
        )
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, f"{TABLE_XPATH}//tr[td]"))
        )

        data = parse_table(driver, TABLE_XPATH)
        save_json("data.json", data)
        print(f"Сохранено записей: {len(data)}")

        for r in data[:3]:
            print(r)

    except TimeoutException:
        # для диагностики сохраним HTML
        with open("page_dump.html", "w", encoding="utf-8") as f:
            f.write(driver.page_source)
        print("Не дождался таблицы/строк. HTML сохранён в page_dump.html")
    finally:
        driver.quit()
