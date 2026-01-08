import os
import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def fetch_rules_index(base_url):
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, 'lxml')
    table = soup.find('table', class_='dataplugin_table')
    
    if not table:
        print("Error: Could not find the rules table.")
        return []

    rows = table.find_all('tr')
    rules_list = []
    
    for row in rows:
        title_cell = row.find('td', class_='عنوان')
        date_cell = row.find('td', class_='ویرایش')
        
        if title_cell and date_cell:
            link_tag = title_cell.find('a')
            if link_tag:
                rules_list.append({
                    "title": link_tag.get_text(strip=True),
                    "url": urljoin(base_url, link_tag['href']),
                    "date": date_cell.get_text(strip=True)
                })
    return rules_list

def fetch_page_body(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'lxml')
        
        main_content = soup.find('main', id='writr__main')
        if not main_content:
            return ""

        for element in main_content.find_all(['div', 'p'], class_=['breadcrumbs', 'page-footer']):
            element.decompose()
            
        return main_content.get_text(separator=' ', strip=True)
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return ""

def save_to_json(data, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def run_collection():
    target_url = "https://ac.sharif.edu/rules/"
    output_path = "../data/raw_content.json"
    
    print(f"Fetching index from: {target_url}")
    rules = fetch_rules_index(target_url)
    
    dataset = []
    print(f"Found {len(rules)} rules. Starting extraction...")
    
    for rule in rules:
        print(f"Processing: {rule['title']}")
        full_text = fetch_page_body(rule['url'])
        dataset.append({
            "title": rule['title'],
            "date": rule['date'],
            "url": rule['url'],
            "content": full_text
        })
    
    save_to_json(dataset, output_path)
    print(f"Success: Saved raw data to {output_path}")

if __name__ == "__main__":
    run_collection()