import os
from datetime import datetime
import json
from typing import List, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, StuffDocumentsChain
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document

# 환경 변수 로드
load_dotenv()

# FastAPI 앱 초기화
app = FastAPI()

# 대화 내용을 저장할 디렉토리 생성
os.makedirs("conversations", exist_ok=True)

# CSV 파일 로드 및 문서 생성
def load_documents(file_name):
    loader = CSVLoader(f"data/{file_name}", encoding='utf-8')
    try:
        documents = loader.load()
    except UnicodeDecodeError:
        loader = CSVLoader(f"data/{file_name}", encoding='cp949')
        documents = loader.load()
    return documents

# 문서 분할
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# 각 제품 카테고리별 벡터 저장소 생성
refrigerator_docs = text_splitter.split_documents(load_documents("refrigerator.csv"))
air_conditioner_docs = text_splitter.split_documents(load_documents("air_conditioner.csv"))
tv_docs = text_splitter.split_documents(load_documents("tv.csv"))

embeddings = OpenAIEmbeddings()

refrigerator_vectorstore = Chroma.from_documents(refrigerator_docs, embeddings, collection_name="refrigerator_collection")
air_conditioner_vectorstore = Chroma.from_documents(air_conditioner_docs, embeddings, collection_name="air_conditioner_collection")
tv_vectorstore = Chroma.from_documents(tv_docs, embeddings, collection_name="television_collection")

# LLM 초기화
llm = ChatOpenAI(temperature=0,model_name="gpt-4o-mini")

# 프롬프트 템플릿 정의
question_prompt_template = """다음 컨텍스트를 사용하여 질문에 답하세요. 
컨텍스트에 관련 정보가 없다면, '죄송합니다. 제가 가진 정보로는 답변드리기 어렵습니다.'라고 대답하세요.
답변은 친절하고 상세하게 해주세요.

컨텍스트: {context}

질문: {question}

답변:"""
QUESTION_PROMPT = PromptTemplate(
    template=question_prompt_template, input_variables=["context", "question"]
)

# 문서 처리를 위한 프롬프트 템플릿
document_prompt = PromptTemplate(
    input_variables=["page_content"],
    template="내용: {page_content}"
)

# LLM 체인 생성
llm_chain = LLMChain(llm=llm, prompt=QUESTION_PROMPT)

# 최종 체인 생성
stuff_chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_variable_name="context",
    document_prompt=document_prompt
)

# 제품 카테고리 분류 함수
def classify_product_category(question: str) -> str:
    classification_prompt = f"""
    다음 질문이 어떤 제품 카테고리에 관한 것인지 분류해주세요. 가능한 카테고리는 '냉장고', '에어컨', 'TV' 입니다.
    질문에 특정 제품이 언급되지 않았다면, 질문의 내용을 바탕으로 가장 관련 있는 카테고리를 추측해주세요.

    질문: {question}

    카테고리:
    """
    response = llm.predict(classification_prompt)
    if '냉장고' in response:
        return 'refrigerator_collection'
    elif '에어컨' in response:
        return 'air_conditioner_collection'
    elif 'TV' in response:
        return 'television_collection'
    else:
        return 'unknown'

# 사용자 대화 내용 저장 함수
def save_conversation(user_id: str, conversation: List[Dict]):
    today = datetime.now().strftime("%Y-%m-%d")
    filename = f"conversations/{user_id}.json"
    
    if os.path.exists(filename):
        with open(filename, "r", encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = {}
    
    if today not in data:
        data[today] = []
    
    data[today].extend(conversation)
    
    with open(filename, "w", encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# 대화 요약 함수
def summarize_conversation(conversation: List[Dict]) -> str:
    summary_prompt = f"다음은 사용자와의 대화 내용입니다. 이 대화를 간단히 요약해주세요:\n\n"
    for message in conversation:
        summary_prompt += f"{'사용자' if message['type'] == 'human' else '시스템'}: {message['content']}\n"
    summary_prompt += "\n요약:"

    summary = llm.predict(summary_prompt)
    return summary

# 입력 모델 정의
class Query(BaseModel):
    user_id: str
    question: str

# 질문 처리를 위한 엔드포인트
@app.post("/ask")
async def ask_question(query: Query):
    try:
        category = classify_product_category(query.question)
        
        if category == 'refrigerator_collection':
            vectorstore = refrigerator_vectorstore
        elif category == 'air_conditioner_collection':
            vectorstore = air_conditioner_vectorstore
        elif category == 'television_collection':
            vectorstore = tv_vectorstore
        else:
            raise HTTPException(status_code=400, detail="질문의 제품 카테고리를 식별할 수 없습니다.")

        docs = vectorstore.similarity_search(query.question, k=3)
        
        result = stuff_chain.run(input_documents=docs, question=query.question)
        
        conversation = [
            {"type": "human", "content": query.question},
            {"type": "ai", "content": result}
        ]
        save_conversation(query.user_id, conversation)
        
        return {"answer": result, "category": category}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 대화 요약을 위한 엔드포인트
@app.post("/summarize")
async def summarize(user_id: str):
    filename = f"conversations/{user_id}.json"
    if not os.path.exists(filename):
        raise HTTPException(status_code=404, detail="사용자의 대화 기록이 없습니다.")
    
    with open(filename, "r", encoding='utf-8') as f:
        data = json.load(f)
    
    all_conversations = []
    for date, conversations in data.items():
        all_conversations.extend(conversations)
    
    summary = summarize_conversation(all_conversations)
    return {"summary": summary}

# 메인 실행 부분
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8989)