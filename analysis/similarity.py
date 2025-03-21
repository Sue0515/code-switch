import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine, euclidean

def calculate_pairwise_metrics(data, embeddings):
    """
    Calculate pairwise cosine similarities and Euclidean distances
    """
    similarity_results = []
    distance_results = []
    
    for i in range(len(data)):
        korean_emb = embeddings['Korean'][i]
        english_emb = embeddings['English'][i]
        ktoe_emb = embeddings['KtoE'][i]
        etok_emb = embeddings['EtoK'][i]
        
        # cosine similarities
        similarity_results.append({
            'Pair_ID': i,
            'Korean_English_Cosine': 1 - cosine(korean_emb, english_emb),
            'KtoE_EtoK_Cosine': 1 - cosine(ktoe_emb, etok_emb),
            'Korean_KtoE_Cosine': 1 - cosine(korean_emb, ktoe_emb),
            'English_EtoK_Cosine': 1 - cosine(english_emb, etok_emb),
            'Korean_EtoK_Cosine': 1 - cosine(korean_emb, etok_emb),
            'English_KtoE_Cosine': 1 - cosine(english_emb, ktoe_emb)
        })
        
        # Euclidean distances
        distance_results.append({
            'Pair_ID': i,
            'Korean_English_Euclidean': euclidean(korean_emb, english_emb),
            'KtoE_EtoK_Euclidean': euclidean(ktoe_emb, etok_emb),
            'Korean_KtoE_Euclidean': euclidean(korean_emb, ktoe_emb),
            'English_EtoK_Euclidean': euclidean(english_emb, etok_emb),
            'Korean_EtoK_Euclidean': euclidean(korean_emb, etok_emb),
            'English_KtoE_Euclidean': euclidean(english_emb, ktoe_emb)
        })
    
    # Convert to DataFrames
    similarity_df = pd.DataFrame(similarity_results)
    distance_df = pd.DataFrame(distance_results)
    
    return similarity_df, distance_df

def analyze_content_similarities(data, embeddings):
    """
    Analyze similarities between different language versions of the same content
    """
    content_similarities = []
    
    for i in range(len(data)):
        ko_en_sim = 1 - cosine(embeddings['Korean'][i], embeddings['English'][i])
        ko_ktoe_sim = 1 - cosine(embeddings['Korean'][i], embeddings['KtoE'][i])
        ko_etok_sim = 1 - cosine(embeddings['Korean'][i], embeddings['EtoK'][i])
        en_ktoe_sim = 1 - cosine(embeddings['English'][i], embeddings['KtoE'][i])
        en_etok_sim = 1 - cosine(embeddings['English'][i], embeddings['EtoK'][i])
        ktoe_etok_sim = 1 - cosine(embeddings['KtoE'][i], embeddings['EtoK'][i])
        
        # Calculate the average similarity for this content
        avg_sim = np.mean([ko_en_sim, ko_ktoe_sim, ko_etok_sim, en_ktoe_sim, en_etok_sim, ktoe_etok_sim])
        
        content_similarities.append({
            'Content_ID': i,
            'English_Text': data[i]['English'][:50] + '...' if len(data[i]['English']) > 50 else data[i]['English'],
            'KO-EN': ko_en_sim,
            'KO-KTOE': ko_ktoe_sim,
            'KO-ETOK': ko_etok_sim,
            'EN-KTOE': en_ktoe_sim,
            'EN-ETOK': en_etok_sim,
            'KTOE-ETOK': ktoe_etok_sim,
            'Average_Similarity': avg_sim
        })
    
    return pd.DataFrame(content_similarities)

def analyze_content_vs_language_effect(data, embeddings):
    """
    Analyze how much content vs. language affects embedding similarity
    """
    language_types = list(embeddings.keys())
    
    # Calculate average similarity within content vs. across content
    within_content_similarities = []
    across_content_similarities = []
    
    for i in range(len(data)):
        for lang1 in language_types:
            for lang2 in language_types:
                if lang1 != lang2:
                    # Within content similarity
                    sim = 1 - cosine(embeddings[lang1][i], embeddings[lang2][i])
                    within_content_similarities.append(sim)
                    
                    # Across content similarities (5 random other contents)
                    for _ in range(5):
                        j = np.random.randint(0, len(data))
                        if j != i:
                            sim = 1 - cosine(embeddings[lang1][i], embeddings[lang2][j])
                            across_content_similarities.append(sim)
    
    return within_content_similarities, across_content_similarities