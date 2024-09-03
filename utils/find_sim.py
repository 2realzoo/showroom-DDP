import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def categorize_response(response, embedding_model):
    # 카테고리와 해당 텍스트의 임베딩을 미리 계산
    categories = {
        'refrigerator_collection': embedding_model.embed_query('냉장고'),
        'air_conditioner_collection': embedding_model.embed_query('에어컨'),
        'television_collection': embedding_model.embed_query('TV')
    }

    # response 임베딩 계산
    response_embedding = embedding_model.embed_query(response.content)

    # 각 카테고리와 response의 코사인 유사도 계산
    similarities = {}
    for category, category_embedding in categories.items():
        similarity = cosine_similarity([response_embedding], [category_embedding])[0][0]
        similarities[category] = similarity

    # 가장 유사도가 높은 카테고리 반환
    best_match = max(similarities, key=similarities.get)

    # 임계값을 설정하여 유사도가 너무 낮은 경우 'unknown' 반환
    if similarities[best_match] < 0.5:  # 임계값은 필요에 따라 조정 가능
        return 'unknown'
    
    return best_match

