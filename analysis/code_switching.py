import pandas as pd
from scipy.spatial.distance import cosine

def analyze_code_switching(data, embeddings):
    """
    Analyze the effect of code-switching on embeddings
    """
    code_switch_analysis = []
    
    for i in range(len(data)):
        # KtoE analysis 
        ktoe_to_ko = 1 - cosine(embeddings['KtoE'][i], embeddings['Korean'][i])
        ktoe_to_en = 1 - cosine(embeddings['KtoE'][i], embeddings['English'][i])
        ktoe_bias = ktoe_to_ko - ktoe_to_en  # Positive means closer to Korean
        
        # EtoK analysis 
        etok_to_ko = 1 - cosine(embeddings['EtoK'][i], embeddings['Korean'][i])
        etok_to_en = 1 - cosine(embeddings['EtoK'][i], embeddings['English'][i])
        etok_bias = etok_to_ko - etok_to_en  # Positive means closer to Korean
        
        code_switch_analysis.append({
            'Content_ID': i,
            'English_Text': data[i]['English'][:50] + '...' if len(data[i]['English']) > 50 else data[i]['English'],
            'KtoE_Korean_Similarity': ktoe_to_ko,
            'KtoE_English_Similarity': ktoe_to_en,
            'KtoE_Bias': ktoe_bias,  # Positive = closer to Korean
            'EtoK_Korean_Similarity': etok_to_ko,
            'EtoK_English_Similarity': etok_to_en,
            'EtoK_Bias': etok_bias  # Positive = closer to Korean
        })
    
    return pd.DataFrame(code_switch_analysis)