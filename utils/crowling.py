import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import os
from config.path import PRODUCT_DATA_FOLDER_PATH

# 페이지 리스트 정의
tv_pages = [
    'https://www.samsung.com/sec/tvs/qled-qde1afxkr-d2c/KQ85QDE1-W2/', 
    'https://www.samsung.com/sec/lifestyletv/the-serif-lsd01afxkr-d2c/KQ65LSD01AFXKR/'
]
refrigerator_pages = [
    'https://www.samsung.com/sec/refrigerators/french-door-rf90dg91114e-d2c/RF90DG9111S9/',
    'https://www.samsung.com/sec/refrigerators/side-by-side-rs84b508115-d2c/RS84B5081SA/', 
    'https://www.samsung.com/sec/refrigerators/side-by-side-rs63r557eb4-d2c/RS63R557EB4/', 
    'https://www.samsung.com/sec/refrigerators/top-mount-freezer-rt31cb5624c3-d2c/RT31CB5624C3/'
]
airconditioner_pages = [
    'https://www.samsung.com/sec/air-conditioners/package-af17dx738bzrt-d2c/AF17DX738BZT/',
    'https://www.samsung.com/sec/air-conditioners/package-af18dx839bzrt-d2c/AF18DX839BZT/',
    'https://www.samsung.com/sec/air-conditioners/package-af19dx838bzrt-d2c/AF19DX838GZT/',
    'https://www.samsung.com/sec/air-conditioners/package-af20dx939bzrt-d2c/AF20DX939BZT/',
    'https://www.samsung.com/sec/air-conditioners/package-af20dx936wfrt-d2c/AF25DX936VFT/',
    'https://www.samsung.com/sec/air-conditioners/package-ar07d9150hzt-d2c/AR07D9150HZT/', 
    'https://www.samsung.com/sec/air-conditioners/package-ar09d9150hzt-d2c/AR09D9150HZT/',
    'https://www.samsung.com/sec/air-conditioners/package-ar11d9150hzt-d2c/AR11D9150HZT/',
    'https://www.samsung.com/sec/air-conditioners/package-ar13d9150hzt-d2c/AR13D9150HZT/',
    'https://www.samsung.com/sec/air-conditioners/package-ar15d9150hzt-d2c/AR15D9150HZT/',
]

categories = {
    "tv": tv_pages,
    "refrigerator": refrigerator_pages,
    "air_conditioner": airconditioner_pages
}

def crawling(url):
    # 웹 드라이버 설정 (Chrome 사용)
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    driver.get(url)
    
    # 페이지 로드 후 '스펙' 버튼이 있는 경우 클릭
    try:
        spec_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.ID, "specDropBtn"))
        )
        spec_button.click()

        # 테이블이 나타날 때까지 기다리기
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CLASS_NAME, "spec-table"))
        )
    except Exception as e:
        print(f"Error clicking the button or waiting for the table: {e}")
    
    # 페이지 소스를 BeautifulSoup으로 파싱
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    
    # 제품 이름 추출
    product_name = soup.select_one('#goodsDetailNm').get_text(strip=True)
    
    # 빈 데이터프레임 생성
    df = pd.DataFrame({
        ('기본정보', '', '제품명'): [product_name],
        ('기본정보', '', '설명'): [''],
        ('기본정보', '', '색상'): [''],
        ('기본정보', '', '사이즈'): ['']
    })
    
    # 색상 정보 추가
    if soup.find('dl', class_='colorchip-group'): 
        color_tags = soup.select('#dropOption-1001-1 > ol > li > label > span')
        colors = ','.join([color_tag.get_text() for color_tag in color_tags])
        df[('기본정보', '', '색상')] = colors

    # 사이즈 정보 추가
    label_selector = 'div.itm-option-choice.droptoggle > dl > dt > span'
    size_value_selector = '.dropList > li > label > span'

    labels = [l.get_text(strip=True) for l in soup.select(label_selector)]
    
    if '사이즈' in labels:
        size_tags = soup.select(size_value_selector)
        sizes = ','.join([size_tag.get_text(strip=True) for size_tag in size_tags if 'cm' in size_tag.get_text()])
        df[('기본정보', '', '사이즈')] = sizes

    # 설명 정보 추가
    tags = soup.find_all(['h2', 'p'], class_='pc-ver')
    contents = ','.join([tag.get_text(strip=True) for tag in tags])
    images = soup.find_all('img', alt=True, class_='obj-m')
    contents += ',' + ','.join([img['alt'] for img in images])
    df[('기본정보','','설명')] = contents
    
    # 스펙 테이블 데이터 추가
    table = soup.find('div', class_='spec-table')
    if table:
        for row in table.find_all('dl'):
            spec_name = row.find('dt').get_text(strip=True)
            specs = row.find_all('li')
            for spec in specs:
                spec_title = spec.find('strong', class_='spec-title').get_text(strip=True) if spec.find('strong', class_='spec-title') else 'N/A'
                spec_desc = spec.find('p', class_='spec-desc').get_text(strip=True) if spec.find('p', class_='spec-desc') else 'N/A'
                if spec_title == '화면크기':
                    continue
                
                # 스펙 데이터를 새로운 열에 추가
                df[('스펙', spec_name, spec_title)] = spec_desc

    # 드라이버 종료
    driver.quit()
    
    return df

def make_data(category_name, pages): 
    combined_df = pd.DataFrame()
    
    for page in pages:
        df = crawling(page)
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    
    file_path = os.path.join(PRODUCT_DATA_FOLDER_PATH, f'{category_name}.csv')
    combined_df.to_csv(file_path, index=False, encoding='utf-8-sig')

# 각 카테고리별로 데이터 생성
for category_name, pages in categories.items():
    make_data(category_name, pages)
