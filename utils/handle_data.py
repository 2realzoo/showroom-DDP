import json
import os
from datetime import datetime
from typing import Dict, List

from langchain_community.document_loaders import CSVLoader


# CSV 파일 로드 및 문서 생성
def load_documents(file_name):
    loader = CSVLoader(f"data/{file_name}", encoding='utf-8-sig')
    try:
        documents = loader.load()
    except UnicodeDecodeError:
        loader = CSVLoader(f"data/{file_name}", encoding='cp949')
        documents = loader.load()
    return documents

# 사용자 대화 내용 저장 함수
def save_conversation(user_id: str, conversation: List[Dict]):
    today = datetime.now().strftime("%Y-%m-%d")
    filename = f"conversations/{user_id}.json"
    
    if os.path.exists(filename):
        with open(filename, "r", encoding='utf-8-sig') as f:
            data = json.load(f)
    else:
        data = {}
    
    if today not in data:
        data[today] = []
    
    data[today].extend(conversation)
    
    with open(filename, "w", encoding='utf-8-sig') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)