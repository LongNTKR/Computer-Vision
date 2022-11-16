from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os
import requests
import time
class CrawlDataWiki(webdriver.Chrome):
    def __init__(self):
        self.string = """Hãy giúp đỡ chúng tôi cập nhật thông tin Đại dịch COVID-19 để mang đến người xem nguồn thông tin chính xác, khách quan và không thiên vị.
Nhớ giữ gìn sức khỏe, thực hiện tốt các biện pháp phòng dịch vì sức khỏe cho bản thân, gia đình và cộng đồng."""
        os.environ['PATH'] += r"C:/SeleniumDrivers"
        super(CrawlDataWiki, self).__init__()

    def search(self, base_url = "https://vi.wikipedia.org"):
        self.get(base_url)

    def get_random_page(self, page_num):
        while True:
            try:
                WebDriverWait(self, 30).until(
                    EC.element_to_be_clickable(
                        (By.ID, "mw-sidebar-button")
                    )
                ).click()
                break
            except:
                continue

        while True:
            try:
                WebDriverWait(self, 30).until(
                    EC.element_to_be_clickable(
                        (By.PARTIAL_LINK_TEXT, "Bài viết ngẫu nhiên")
                    )
                ).click()
                break
            except:
                continue

        while True:
            try:
                elements = WebDriverWait(self, 30).until(
                    EC.presence_of_all_elements_located(
                        (By.CSS_SELECTOR, "div p")
                    )
                )
                break
            except:
                continue

        with open(r'd:\code\python\selenium\crawl_data_wiki\crawl_data_wiki\data.txt', 'a', encoding='utf-8') as file:
            for element in elements:
                if len(element.text) > 0 and element.text != self.string:
                    file.writelines(element.text)
            file.close()

driver = CrawlDataWiki()
driver.search()
count = 0
while True:
    driver.get_random_page(count)
    count += 1
    print(count)

