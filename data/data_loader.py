import json

def load_data(filepath):
    """
    Load data from JSON file.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def extract_sentences(data):
    """
    Extract sentences from data dictionary.
    """
    korean_sentences = [item['Korean'] for item in data]
    english_sentences = [item['English'] for item in data]
    ktoe_sentences = [item['KtoE'] for item in data]
    etok_sentences = [item['EtoK'] for item in data]
    
    return {
        'Korean': korean_sentences,
        'English': english_sentences,
        'KtoE': ktoe_sentences,
        'EtoK': etok_sentences
    }