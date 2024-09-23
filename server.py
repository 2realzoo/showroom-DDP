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

model = Llm_Model(model_name='gpt')

class Query(BaseModel):
    user_id: str
    question: str
    area_size: str
    housemate_num: str

@app.post("/ask")
async def ask_question(query: Query):
    try:
        with get_openai_callback() as cb:
            result = model.process_question(query.user_id, query.question, query.area_size, query.housemate_num)
        
        conversation = [
            {"type": "human", "content": query.question},
            {"type": "ai", "content": result['answer']}
        ]
        save_conversation(query.user_id, conversation)
        
        return JSONResponse(content=result)
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8282)