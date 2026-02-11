import os
import requests
from bs4 import BeautifulSoup
import urllib.parse
import re
import uuid
import json
from tqdm import tqdm

BASE_URL = "https://ac.sharif.edu"
MAIN_RULES_URL = "https://ac.sharif.edu/rules/"

def get_rule_links():
    response = requests.get(MAIN_RULES_URL)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    
    links = []
    table = soup.find('table', class_='inline dataplugin_table')
    rows = table.find_all('tr')[1:]
    
    for row in rows:
        cols = row.find_all('td')
        if len(cols) == 2:
            a_tag = cols[0].find('a')
            if a_tag:
                href = a_tag['href']
                full_url = urllib.parse.urljoin(BASE_URL, href)
                rule_title = a_tag.text.strip()
                rule_date = cols[1].text.strip()
                
                links.append({
                    'url': full_url,
                    'title': rule_title,
                    'date': rule_date
                })
    return links


def parse_table_to_markdown(table_tag):
    markdown = "\n"
    rows = table_tag.find_all('tr')
    if not rows:
        return ""
    
    for i, row in enumerate(rows):
        cols = row.find_all(['th', 'td'])
        row_data = [re.sub(r'\s+', ' ', col.get_text(strip=True)) for col in cols]
        
        if not any(row_data):
            continue
            
        markdown += "| " + " | ".join(row_data) + " |\n"
        
        if i == 0:
            markdown += "|" + "|".join(["---"] * len(cols)) + "|\n"
            
    return markdown + "\n"


def process_rule_page(rule_info):
    response = requests.get(rule_info['url'])
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    
    main_content = soup.find('main', id='writr__main')
    if not main_content:
        return []

    elements = main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li', 'table'])
    
    chunks = []
    current_chunk_text = ""
    current_parent_section = "General"
    current_section_title = "General"
    
    split_pattern = r'^\s*(ماده\s*\d+|مقدمه|[الف-ی]\s*[:\)]?)'

    for el in elements:
        if el.name != 'table' and el.find_parent('table'):
            continue
            
        if el.name == 'table':
            current_chunk_text += parse_table_to_markdown(el)
            continue

        text = el.get_text(separator=' ', strip=True)
        if not text:
            continue

        is_header = el.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']
        is_strong_trigger = False
        strong_text = ""
        
        if not is_header:
            strong_tag = el.find(['strong', 'b'])
            if strong_tag:
                strong_text = strong_tag.get_text(strip=True)
                if text.startswith(strong_text) and re.match(split_pattern, text):
                    is_strong_trigger = True

        if is_header or is_strong_trigger:
            if current_chunk_text.strip():
                chunks.append({
                    "id": str(uuid.uuid4()),
                    "rule_title": rule_info['title'],
                    "rule_url": rule_info['url'],
                    "rule_date": rule_info['date'],
                    "parent_section": current_parent_section,
                    "section_title": current_section_title,
                    "content": current_chunk_text.strip()
                })
            
            if is_header:
                current_parent_section = text
                current_section_title = text
                current_chunk_text = text + "\n"
            else:
                current_section_title = strong_text
                current_chunk_text = text + "\n"
        
        else:
            if el.name == 'li':
                current_chunk_text += "- " + text + "\n"
            else:
                current_chunk_text += text + "\n"
    
    if current_chunk_text.strip():
        chunks.append({
            "id": str(uuid.uuid4()),
            "rule_title": rule_info['title'],
            "rule_url": rule_info['url'],
            "rule_date": rule_info['date'],
            "parent_section": current_parent_section,
            "section_title": current_section_title,
            "content": current_chunk_text.strip()
        })
        
    return chunks


def main():
    rules = get_rule_links()
    print(f"{len(rules)} rules found. Processing...")
    
    all_chunks = []
    
    for rule in tqdm(rules, desc="Processing rules"):
        try:
            chunks = process_rule_page(rule)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"Error processing {rule['url']}: {e}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)

    output_path = os.path.join(data_dir, "sharif_rules_chunks.json")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=4)
    
    print(f"Done. Total chunks: {len(all_chunks)}")
    print(f"Saved to '{output_path}'")


if __name__ == "__main__":
    main()
