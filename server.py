import json
import os

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from langchain.callbacks import get_openai_callback
from pydantic import BaseModel

from model.llm_model import Llm_Model
from utils.handle_data import save_conversation

app = FastAPI()

# TODO:회원 별 instance 생성
model = Llm_Model(model_name='gpt')

class Query(BaseModel):
    user_id: str
    question: str
    area_size: str
    housemate_num: str

@app.post("/ask")
async def ask_question(query: Query):
    try:
        category = model.classify_product_category(query.question)
        
        if category == 'refrigerator_collection':
            vectorStore = model.refrigerator_vectorStore
        elif category == 'air_conditioner_collection':
            vectorStore = model.air_conditioner_vectorStore
        elif category == 'television_collection':
            vectorStore = model.tv_vectorStore
        else:
            return JSONResponse(
                status_code=400,
                content={"answer": "질문의 제품 카테고리를 식별할 수 없습니다."}
            )

        docs = vectorStore.similarity_search(query.question, k=5)
        
        with get_openai_callback() as cb:
            result = model.chain.run(input_documents=docs, question=query.question)
            # print(f"Total Tokens: {cb.total_tokens}")
            # print(f"Prompt Tokens: {cb.prompt_tokens}")
            # print(f"Completion Tokens: {cb.completion_tokens}")
            # print(f"Total Cost (USD): ${cb.total_cost}")
            # print(result)
        limited_answer = model.limit_tokens(result)
        
        conversation = [
            {"type": "human", "content": query.question},
            {"type": "ai", "content": limited_answer}
        ]
        save_conversation(query.user_id, conversation)
        model.memory.chat_memory.add_user_message(query.question)
        model.memory.chat_memory.add_ai_message(limited_answer)
        
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

@app.post("/summarize")
async def summarize(user_id: str):
    filename = f"conversations/{user_id}.json"
    if not os.path.exists(filename):
        return JSONResponse(
            status_code=404,
            content={"error": "사용자의 대화 기록이 없습니다."}
        )
    
    with open(filename, "r", encoding='utf-8-sig') as f:
        data = json.load(f)
    
    all_conversations = []
    for date, conversations in data.items():
        all_conversations.extend(conversations)
    
    summary = model.summarize_conversation(all_conversations)
    return JSONResponse(
        content={"summary": summary}
    )

# 메인 실행 부분
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8282)