import os
import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def fetch_rules_list(base_url):
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, 'lxml')
    
    # تلاش برای پیدا کردن جدول با کلاس مربوطه
    table = soup.find('table', class_='dataplugin_table')
    
    if not table:
        print("خطا: جدول آیین‌نامه‌ها پیدا نشد!")
        return []

    # مستقیم سراغ تگ‌های tr می‌رویم (چه داخل tbody باشند چه نباشند)
    rows = table.find_all('tr')
    
    rules = []
    for row in rows:
        # کلاس‌ها را طبق HTML ارسالی شما چک می‌کنیم
        title_cell = row.find('td', class_='عنوان')
        date_cell = row.find('td', class_='ویرایش')
        
        if title_cell and date_cell:
            link = title_cell.find('a')
            if link:
                rules.append({
                    "title": link.get_text(strip=True),
                    "url": urljoin(base_url, link['href']),
                    "date": date_cell.get_text(strip=True)
                })
    
    return rules

if __name__ == "__main__":
    URL = "https://ac.sharif.edu/rules/"
    all_rules = fetch_rules_list(URL)
    
    if all_rules:
        print(f"تعداد {len(all_rules)} آیین‌نامه پیدا شد.")
        for r in all_rules[:3]:
            print(f"Title: {r['title']} | Date: {r['date']}")
    else:
        print("هیچ داده‌ای استخراج نشد.")