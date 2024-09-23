import os
from typing import Dict, List
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

import json 
from dotenv import load_dotenv
from langchain.chains import LLMChain, MapReduceDocumentsChain, StuffDocumentsChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma

from model.model_zoo import gpt_embedding, gpt_model, llama_embedding, llama_model
from utils.find_sim import categorize_response
from utils.handle_data import load_documents, save_conversation

load_dotenv()

class UserMemory:
    def __init__(self):
        self.conversation_history = []
        self.current_category = None
        self.memory = ConversationBufferMemory(memory_key="history", max_token_limit=100)

    def add_to_history(self, question: str, answer: str):
        self.conversation_history.append({"question": question, "answer": answer})
        if len(self.conversation_history) > 5:  # 최근 5개의 대화만 유지
            self.conversation_history.pop(0)

class Llm_Model():
    def __init__(self, model_name='gpt'):
        self.model = gpt_model if model_name == 'gpt' else llama_model
        self.embedding = gpt_embedding if model_name == 'gpt' else llama_embedding
        
        persist_directory = 'samsung.db'
        self._initialize_vector_stores(persist_directory)
        
        self._setup_chain()
        
        self.user_memories = {}
        self.memory = ConversationBufferMemory(memory_key="history", max_token_limit=100)

    def _initialize_vector_stores(self, persist_directory):
        if os.path.exists(persist_directory):
            self.refrigerator_vectorStore = Chroma(collection_name="refrigerator_collection", persist_directory=persist_directory, embedding_function=self.embedding)
            self.air_conditioner_vectorStore = Chroma(collection_name="air_conditioner_collection", persist_directory=persist_directory, embedding_function=self.embedding)
            self.tv_vectorStore = Chroma(collection_name="tv_collection", persist_directory=persist_directory, embedding_function=self.embedding)
        else:
            self._create_vector_stores(persist_directory)

        self.refrigerator_retriever = self.refrigerator_vectorStore.as_retriever(search_kwargs={'k': 2})
        self.air_conditioner_retriever = self.air_conditioner_vectorStore.as_retriever(search_kwargs={'k': 2})
        self.tv_retriever = self.tv_vectorStore.as_retriever(search_kwargs={'k': 2})

    def _create_vector_stores(self, persist_directory):
        text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        refrigerator_docs = text_splitter.split_documents(load_documents("refrigerator.csv"))
        air_conditioner_docs = text_splitter.split_documents(load_documents("air_conditioner.csv"))
        tv_docs = text_splitter.split_documents(load_documents("tv.csv"))
        
        self.refrigerator_vectorStore = Chroma.from_documents(
            documents=refrigerator_docs,
            embedding=self.embedding,
            collection_name="refrigerator_collection",
            persist_directory=persist_directory
        )
        self.air_conditioner_vectorStore = Chroma.from_documents(
            documents=air_conditioner_docs,
            embedding=self.embedding,
            collection_name="air_conditioner_collection",
            persist_directory=persist_directory
        )
        self.tv_vectorStore = Chroma.from_documents(
            documents=tv_docs,
            embedding=self.embedding,
            collection_name="tv_collection",
            persist_directory=persist_directory
        )
        
        self.refrigerator_vectorStore.persist()
        self.air_conditioner_vectorStore.persist()
        self.tv_vectorStore.persist()

    def _setup_chain(self):
        self.map_template = """
        다음 문서에서 질문에 관련된 정보를 추출하세요:
        {document}
        
        질문: {question}
        
        관련 정보:"""
        self.map_prompt = PromptTemplate.from_template(self.map_template)
        
        self.map_chain = LLMChain(llm=self.model, prompt=self.map_prompt)
        
        self.reduce_template = """다음은 질문에 대한 관련 정보 조각들입니다. 이 정보를 바탕으로 질문에 대한 최종 답변을 작성하세요.
        정보 조각들:
        {context}
        
        질문: {question}
        
        주의사항:
        1. 답변은 친절하고 상세하게 해주세요.
        2. 정보가 부족하거나 관련이 없다면, '죄송합니다. 제가 가진 정보로는 답변드리기 어렵습니다.'라고 대답하세요.
        3. 답변에 불확실한 내용이 있다면 그 부분을 명시해주세요.
        5. 제품의 모델명을 같이 알려주세요.
        
        답변:
        """
        self.reduce_prompt = PromptTemplate.from_template(self.reduce_template)
        
        self.reduce_chain = LLMChain(llm=self.model, prompt=self.reduce_prompt)
        
        combine_documents_chain = StuffDocumentsChain(
            llm_chain=self.reduce_chain,
            document_variable_name="context"
        )
        
        self.chain = MapReduceDocumentsChain(
            llm_chain=self.map_chain,
            reduce_documents_chain=combine_documents_chain,
            document_variable_name="document",
            return_intermediate_steps=False
        )

    def classify_product_category(self, user_id: str, question: str) -> str:
        user_memory = self.user_memories.get(user_id)
        if user_memory and user_memory.current_category and self.is_followup_question(question):
            return user_memory.current_category

        classification_prompt = f"""
        다음 질문이 어떤 제품 카테고리에 관한 것인지 분류해주세요. 가능한 카테고리는 '냉장고', '에어컨', 'TV' 입니다.
        질문에 특정 제품이 언급되지 않았다면, 질문의 내용을 바탕으로 가장 관련 있는 카테고리를 추측해주세요.

        이전 대화 내역:
        {self.get_conversation_history(user_id)}

        질문: {question}

        카테고리:
        """
        response = self.model.invoke(classification_prompt)
        
        category = categorize_response(response, self.embedding)
        if category not in ['냉장고', '에어컨', 'TV']:
            category = '냉장고'  # 기본값으로 '냉장고' 설정
        if user_memory:
            user_memory.current_category = category
        return category

    def get_retriever_for_category(self, category: str):
        if category == "냉장고":
            return self.refrigerator_retriever
        elif category == "에어컨":
            return self.air_conditioner_retriever
        elif category == "TV":
            return self.tv_retriever
        else:
            return self.refrigerator_retriever  # 기본값으로 냉장고 retriever 반환

    def is_followup_question(self, question: str) -> bool:
        followup_keywords = ["그", "이", "저", "해당", "이전"]
        return any(keyword in question for keyword in followup_keywords)

    def get_conversation_history(self, user_id: str) -> str:
        user_memory = self.user_memories.get(user_id)
        if not user_memory:
            return ""
        return "\n".join([f"Q: {item['question']}\nA: {item['answer']}" for item in user_memory.conversation_history])

    def limit_tokens(self, text, max_tokens=6000):
        tokens = text.split()
        if len(tokens) <= max_tokens:
            return text
        return ' '.join(tokens[:max_tokens])
    
    def summarize_conversation(self, user_id: str) -> str:
        user_memory = self.user_memories.get(user_id)
        if not user_memory:
            return "해당 사용자의 대화 기록이 없습니다."

        summary_prompt = f"다음은 사용자와의 대화 내용입니다. 이 대화를 간단히 요약해주세요:\n\n"
        for item in user_memory.conversation_history:
            summary_prompt += f"사용자: {item['question']}\n시스템: {item['answer']}\n"
        summary_prompt += "\n요약:"

        summary = self.model.predict(summary_prompt)
        return summary
    
    def recommend_product(self, user_id: str, question: str, area_size: str, housemate_num: str) -> Dict[str, str]:
        try:
            category = self.classify_product_category(user_id, question)
            
            retriever = self.get_retriever_for_category(category)
            docs = retriever.get_relevant_documents(question)
            
            recommendation_prompt = f"""
            다음 정보를 바탕으로 가장 적합한 제품을 추천해주세요:
            카테고리: {category}
            사용자 질문: {question}
            면적: {area_size}
            가구 구성원 수: {housemate_num}
            
            제품 정보:
            {' '.join([doc.page_content for doc in docs])}
            
            다음 형식으로 추천을 제공해주세요:
            모델명: [추천 모델명]
            추천이유 : [추천 이유]
            특징 1: [특징 1 설명]
            특징 2: [특징 2 설명]
            특징 3: [특징 3 설명]
            
            추천이유는 사용자의 조건에 따른 추천 이유를 한 줄로 설명해주세요.
            각 특징은 모델의 주요 장점을 간결하게 한 줄로 설명해주세요.
            반드시 각 줄 끝에 '\\n'을 추가하여 줄바꿈을 명시해주세요.
            """
            
            response = self.model.invoke(recommendation_prompt)
            
            # AIMessage 객체 처리
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
            
            print(f"모델 응답: {response_text}")  # 디버깅을 위해 응답 출력
            
            return self.generate_fallback_response(response_text)
        except Exception as e:
            print(f"제품 추천 중 오류 발생: {str(e)}")
            return {
                "model": "오류",
                "answer": "죄송합니다. 제품을 추천하는 동안 오류가 발생했습니다. 다시 시도해 주세요."
            }

    def generate_fallback_response(self, response: str) -> Dict[str, str]:
        try:
            lines = response.split('\n')
            model_name = "알 수 없음"
            features = []
            
            for line in lines:
                if "모델명:" in line:
                    model_name = line.split(":", 1)[-1].strip().rstrip('\\n')  # '\\n' 제거
                elif "특징" in line:
                    feature = line.split(":", 1)[-1].strip().rstrip('\\n')  # '\\n' 제거
                    if feature:
                        features.append(feature)
            
            if model_name != "알 수 없음" and features:
                answer = f"모델명: {model_name}\n"
                answer += "\n".join([f"특징 {i+1}: {feature}" for i, feature in enumerate(features[:3])])
                return {"model": model_name, "answer": answer}
            else:
                # 모델명과 특징을 찾지 못한 경우, 전체 응답을 사용
                return {
                    "model": "추천 모델",
                    "answer": f"추천 내용:\n{response.rstrip('\\n')}"  # '\\n' 제거
                }
        except Exception as e:
            print(f"대체 응답 생성 중 오류 발생: {str(e)}")
            return {
                "model": "알 수 없음",
                "answer": f"죄송합니다. 적절한 추천을 생성하는 데 문제가 발생했습니다. 모델의 응답:\n{response.rstrip('\\n')}"  # '\\n' 제거
            }

    def process_question(self, user_id: str, question: str, area_size: str, housemate_num: str) -> Dict[str, str]:
        if user_id not in self.user_memories:
            self.user_memories[user_id] = UserMemory()

        recommendation = self.recommend_product(user_id, question, area_size, housemate_num)
        
        self.user_memories[user_id].add_to_history(question, recommendation['answer'])
        self.memory.chat_memory.add_user_message(question)
        self.memory.chat_memory.add_ai_message(recommendation['answer'])
        
        return recommendation