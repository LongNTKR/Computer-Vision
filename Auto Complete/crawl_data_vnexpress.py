from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os
import requests
import time
class CrawlDataVNEXPRESS(webdriver.Chrome):
    def __init__(self):
        self.class_names = ['thoisu', 'gocnhin', 'thegioi', 'podcasts', 'kinhdoanh' , 'khoahoc',
              'giaitri', 'thethao', 'phapluat', 'giaoduc', 'suckhoe', 'doisong', 'dulich',
              'sohoa', 'xe', 'ykien', 'tamsu']
        self.articles = []
        self.tmp_articles = []
        os.environ['PATH'] += r"C:/SeleniumDrivers"
        super(CrawlDataVNEXPRESS, self).__init__()

    def get_class_names(self):
        return self.class_names

    def get_articles(self):
        return self.articles

    def search(self, base_url = "https://vnexpress.net/"):
        self.get(base_url)

    def click_all_button(self, locator):
        while True:
            try:
                WebDriverWait(self, 30).until(
                    EC.element_to_be_clickable(locator)
                ).click()
                break
            except:
                continue

    def get_href(self):
        href = []
        for element in self.find_elements(By.CSS_SELECTOR, 'article'):
            try:
                href.append(element.find_element(By.CSS_SELECTOR, 'h3').find_element(By.CSS_SELECTOR, 'a').get_attribute('href'))
            except:
                continue
        return href

    def scroll_to_end(self):
        last_height = self.execute_script("return document.body.scrollHeight")
        scroll_count = 0
        while True:
            self.execute_script("window.scrollTo(0, document.body.scrollHeight)")
            time.sleep(1)
            new_height = self.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                self.tmp_articles += self.get_href()
                if len(self.tmp_articles) > 100:
                    break
                try:
                    self.find_element(By.CLASS_NAME, 'btn-page.next-page ').click()
                except:
                    break
            elif scroll_count == 100:
                self.articles += self.get_href()
                break
            last_height = new_height
            scroll_count += 1
        self.articles = self.tmp_articles
        self.tmp_articles = []
    
    def get_data(self):
        elements = self.find_element(By.CSS_SELECTOR, 'article').find_elements(By.CSS_SELECTOR, 'p')
        with open(r'data.txt', 'a', encoding='utf-8') as file:
            for element in elements:
                if len(element.text) > 0:
                    file.writelines(element.text)
            file.close()

driver = CrawlDataVNEXPRESS()

driver.search()
driver.maximize_window()
for class_name in driver.get_class_names():
    driver.click_all_button((By.CLASS_NAME, 'all-menu.has_transition'))
    driver.search(driver.find_element(By.CLASS_NAME, class_name).find_element(By.CSS_SELECTOR, 'a').get_attribute('href'))
    driver.scroll_to_end()
    with open(r'href.txt', 'a', encoding='utf-8') as file:
        file.write(class_name + '\n')
        for href in driver.get_articles():
            file.write(href + '\n')
        file.close()

with open(r'href.txt', 'r+', encoding='utf-8') as file:
    data = file.readlines()

for url in data:
    try:
        driver.search(url)
        driver.get_data()
    except:
        continue
time.sleep(10)



