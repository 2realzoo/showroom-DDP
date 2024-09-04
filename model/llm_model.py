from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from transformers import PreTrainedTokenizerFast,GPT2LMHeadModel
from langchain.text_splitter import CharacterTextSplitter
from utils.handle_data import load_documents, save_conversation
from model.model_zoo import gpt_model, gpt_embedding, llama_model, llama_embedding
from langchain.chains import LLMChain, StuffDocumentsChain, MapReduceDocumentsChain
from langchain.prompts import PromptTemplate
from utils.find_sim import categorize_response
from typing import List, Dict
import os
from dotenv import load_dotenv

load_dotenv()

class Llm_Model():
    def __init__(self, model_name='gpt'):
        self.model = gpt_model if model_name == 'gpt' else llama_model
        self.embedding = gpt_embedding if model_name == 'gpt' else llama_embedding
        
        # RAG 설정: CSV 로더, 텍스트 분할기, 임베딩, 벡터 저장소 설정
        persist_directory = 'samsung.db'
        if os.path.exists(persist_directory):
            self.refrigerator_vectorStore = Chroma(collection_name="refrigerator_collection", persist_directory=persist_directory, embedding_function=self.embedding)
            self.air_conditioner_vectorStore = Chroma(collection_name="air_conditioner_collection", persist_directory=persist_directory, embedding_function=self.embedding)
            self.tv_vectorStore = Chroma(collection_name="tv_collection", persist_directory=persist_directory, embedding_function=self.embedding)
        else:
            # 각 제품 카테고리별 벡터 저장소 생성
            text_splitter = CharacterTextSplitter(chunk_size=10000, chunk_overlap=500)
            refrigerator_docs = text_splitter.split_documents(load_documents("refrigerator.csv"))
            air_conditioner_docs = text_splitter.split_documents(load_documents("air_conditioner.csv"))
            tv_docs = text_splitter.split_documents(load_documents("tv.csv"))
            
            embedding = gpt_embedding
            
            self.refrigerator_vectorStore = Chroma.from_documents(
                documents=refrigerator_docs,
                embedding=embedding,
                collection_name="refrigerator_collection",
                persist_directory=persist_directory
            )
            self.air_conditioner_vectorStore = Chroma.from_documents(
                documents=air_conditioner_docs,
                embedding=embedding,
                collection_name="air_conditioner_collection",
                persist_directory=persist_directory
            )
            self.tv_vectorStore = Chroma.from_documents(
                documents=tv_docs,
                embedding=embedding,
                collection_name="tv_collection",
                persist_directory=persist_directory
            )
            
            self.refrigerator_vectorStore.persist()
            self.air_conditioner_vectorStore.persist()
            self.tv_vectorStore.persist()
            
            self.refrigerator_retriever = self.refrigerator_vectorStore.as_retriever(search_kwargs={'k': 3})
            self.air_conditioner_retriever = self.air_conditioner_vectorStore.as_retriever(search_kwargs={'k': 3})
            self.tv_retriever = self.tv_vectorStore.as_retriever(search_kwargs={'k': 3})
        
        
        self.map_template = """
        다음 문서에서 질문에 관련된 정보를 추출하세요:
        {document}
        
        질문: {question}
        
        관련 정보:"""
        self.map_prompt = PromptTemplate.from_template(self.map_template)
        
        # 맵 체인
        self.map_chain = LLMChain(llm=self.model, prompt=self.map_prompt)
        
        # 리듀스 프롬프트
        self.reduce_template = """다음은 질문에 대한 관련 정보 조각들입니다. 이 정보를 바탕으로 질문에 대한 최종 답변을 작성하세요.
        정보 조각들:
        {context}
        
        질문: {question}
        
        주의사항:
        1. 답변은 친절하고 상세하게 해주세요.
        2. 정보가 부족하거나 관련이 없다면, '죄송합니다. 제가 가진 정보로는 답변드리기 어렵습니다.'라고 대답하세요.
        3. 답변에 불확실한 내용이 있다면 그 부분을 명시해주세요.
        4. 제품의 구체적인 가격 정보는 언급하지 마세요.
        5. 가능하면 제품의 모델명을 같이 알려주세요.
        
        답변:
        """
        self.reduce_prompt = PromptTemplate.from_template(self.reduce_template)
        
        # 리듀스 체인
        self.reduce_chain = LLMChain(llm=self.model, prompt=self.reduce_prompt)
        
        # 컴바인 독스 체인
        combine_documents_chain = StuffDocumentsChain(
            llm_chain=self.reduce_chain,
            document_variable_name="context"
        )
        
        # 최종 맵리듀스 체인
        self.chain = MapReduceDocumentsChain(
            llm_chain=self.map_chain,
            reduce_documents_chain=combine_documents_chain,
            document_variable_name="document",
            return_intermediate_steps=False
        )
        
        self.memory = ConversationBufferMemory(memory_key="history", max_token_limit=100)

    def classify_product_category(self, question: str) -> str:
        classification_prompt = f"""
        다음 질문이 어떤 제품 카테고리에 관한 것인지 분류해주세요. 가능한 카테고리는 '냉장고', '에어컨', 'TV' 입니다.
        질문에 특정 제품이 언급되지 않았다면, 질문의 내용을 바탕으로 가장 관련 있는 카테고리를 추측해주세요.

        질문: {question}

        카테고리:
        """
        response = self.model.invoke(classification_prompt)
        
        return categorize_response(response, self.embedding)
    
    # 토큰 수 제한 함수
    def limit_tokens(self, text, max_tokens=10000):
        tokens = text.split()
        if len(tokens) <= max_tokens:
            return text
        return ' '.join(tokens[:max_tokens])
    
    # 대화 요약 함수
    def summarize_conversation(self, conversation: List[Dict]) -> str:
        summary_prompt = f"다음은 사용자와의 대화 내용입니다. 이 대화를 간단히 요약해주세요:\n\n"
        for message in conversation:
            summary_prompt += f"{'사용자' if message['type'] == 'human' else '시스템'}: {message['content']}\n"
        summary_prompt += "\n요약:"

        summary = self.model.predict(summary_prompt)
        return summary
      

def main():
    pass


if __name__ == "__main__":
    main()
 