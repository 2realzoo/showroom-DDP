import os
from datetime import datetime
import json
from typing import List, Dict
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, StuffDocumentsChain, MapReduceChain
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import get_openai_callback
from langchain.chains import MapReduceDocumentsChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter

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
text_splitter = CharacterTextSplitter(chunk_size=10000, chunk_overlap=500)

# 각 제품 카테고리별 벡터 저장소 생성
refrigerator_docs = text_splitter.split_documents(load_documents("refrigerator.csv"))
air_conditioner_docs = text_splitter.split_documents(load_documents("air_conditioner.csv"))
tv_docs = text_splitter.split_documents(load_documents("tv.csv"))

embeddings = OpenAIEmbeddings()

refrigerator_vectorstore = Chroma.from_documents(refrigerator_docs, embeddings, collection_name="refrigerator_collection")
air_conditioner_vectorstore = Chroma.from_documents(air_conditioner_docs, embeddings, collection_name="air_conditioner_collection")
tv_vectorstore = Chroma.from_documents(tv_docs, embeddings, collection_name="television_collection")

# LLM 초기화
llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")

# 프롬프트 템플릿 정의
map_prompt_template = """다음 문서에서 질문에 관련된 정보를 추출하세요:
{document}

관련 정보:"""
MAP_PROMPT = PromptTemplate(template=map_prompt_template, input_variables=["document"])

reduce_prompt_template = """다음은 질문에 대한 관련 정보 조각들입니다. 이 정보를 바탕으로 질문에 대한 최종 답변을 작성하세요.
정보 조각들:
{context}

대화 기록: {chat_history}

질문: {question}

주의사항:
1. 답변은 친절하고 상세하게 해주세요.
2. 정보가 부족하거나 관련이 없다면, '죄송합니다. 제가 가진 정보로는 답변드리기 어렵습니다.'라고 대답하세요.
3. 답변에 불확실한 내용이 있다면 그 부분을 명시해주세요.
4. 제품의 구체적인 모델명이나 가격 정보는 언급하지 마세요.

답변:"""
REDUCE_PROMPT = PromptTemplate(
    template=reduce_prompt_template,
    input_variables=["context", "chat_history", "question"]
)

# 메모리 초기화
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 맵리듀스 체인 생성
def create_chain(vectorstore):
    # 맵 프롬프트
    map_template = """다음 문서에서 질문에 관련된 정보를 추출하세요:
    {document}
    
    질문: {question}
    
    관련 정보:"""
    map_prompt = PromptTemplate.from_template(map_template)
    
    # 맵 체인
    map_chain = LLMChain(llm=llm, prompt=map_prompt)
    
    # 리듀스 프롬프트
    reduce_template = """다음은 질문에 대한 관련 정보 조각들입니다. 이 정보를 바탕으로 질문에 대한 최종 답변을 작성하세요.
    정보 조각들:
    {context}
    
    질문: {question}
    
    주의사항:
    1. 답변은 친절하고 상세하게 해주세요.
    2. 정보가 부족하거나 관련이 없다면, '죄송합니다. 제가 가진 정보로는 답변드리기 어렵습니다.'라고 대답하세요.
    3. 답변에 불확실한 내용이 있다면 그 부분을 명시해주세요.
    4. 제품의 구체적인 가격 정보는 언급하지 마세요.
    5. 가능하면 제품의 모델명을 같이 알려주세요.
    
    답변:"""
    reduce_prompt = PromptTemplate.from_template(reduce_template)
    
    # 리듀스 체인
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)
    
    # 컴바인 독스 체인
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain,
        document_variable_name="context"
    )
    
    # 최종 맵리듀스 체인
    return MapReduceDocumentsChain(
        llm_chain=map_chain,
        reduce_documents_chain=combine_documents_chain,
        document_variable_name="document",
        return_intermediate_steps=False
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

# 토큰 수 제한 함수
def limit_tokens(text, max_tokens=10000):
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text
    return ' '.join(tokens[:max_tokens])

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
            return JSONResponse(
                status_code=400,
                content={"answer": "질문의 제품 카테고리를 식별할 수 없습니다."}
            )

        docs = vectorstore.similarity_search(query.question, k=5)
        chain = create_chain(vectorstore)
        
        with get_openai_callback() as cb:
            result = chain.run(input_documents=docs, question=query.question)
            print(f"Total Tokens: {cb.total_tokens}")
            print(f"Prompt Tokens: {cb.prompt_tokens}")
            print(f"Completion Tokens: {cb.completion_tokens}")
            print(f"Total Cost (USD): ${cb.total_cost}")
        
        limited_answer = limit_tokens(result)
        
        conversation = [
            {"type": "human", "content": query.question},
            {"type": "ai", "content": limited_answer}
        ]
        save_conversation(query.user_id, conversation)
        memory.chat_memory.add_user_message(query.question)
        memory.chat_memory.add_ai_message(limited_answer)
        
        return JSONResponse(
            content={
                "answer": limited_answer
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

# 대화 요약을 위한 엔드포인트
@app.post("/summarize")
async def summarize(user_id: str):
    filename = f"conversations/{user_id}.json"
    if not os.path.exists(filename):
        return JSONResponse(
            status_code=404,
            content={"error": "사용자의 대화 기록이 없습니다."}
        )
    
    with open(filename, "r", encoding='utf-8') as f:
        data = json.load(f)
    
    all_conversations = []
    for date, conversations in data.items():
        all_conversations.extend(conversations)
    
    summary = summarize_conversation(all_conversations)
    return JSONResponse(
        content={"summary": summary}
    )

# 메인 실행 부분
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8989)