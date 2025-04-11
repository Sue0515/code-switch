import os
import json
import time
import requests
from typing import List, Dict, Any, Optional, Union
from tqdm import tqdm
import logging

from .document import Document

logger = logging.getLogger(__name__)

class WikipediaRetriever:
    
    def __init__(self, langs=["en", "ko"]):

        self.langs = langs
        
    def search(
        self, 
        query: str, 
        lang: str = "en", 
        limit: int = 10, 
        max_retries: int = 3, 
        retry_delay: float = 1.0
    ) -> List[Document]:

        url = f"https://{lang}.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            "srprop": "snippet|titlesnippet"
        }
        
        # Execute with retries
        retries = 0
        while retries < max_retries:
            try:
                response = requests.get(url, params=params, timeout=10)
                data = response.json()
                break
            except (requests.RequestException, json.JSONDecodeError) as e:
                retries += 1
                logger.warning(f"Error searching Wikipedia in {lang} (attempt {retries}/{max_retries}): {e}")
                if retries < max_retries:
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Failed to search Wikipedia in {lang} after {max_retries} attempts")
                    return []
        
        # Extract search results
        articles = []
        for item in data.get("query", {}).get("search", []):
            # Clean content (remove HTML tags)
            content = item["snippet"].replace("<span class=\"searchmatch\">", "").replace("</span>", "")
            
            # Create document ID
            doc_id = f"wiki_{lang}_{item['pageid']}"
            
            # Determine language name
            if lang == "en":
                language = "english"
            elif lang == "ko":
                language = "korean"
            else:
                language = lang
            
            # Create document
            document = Document(
                id=doc_id,
                content=content,
                title=item["title"],
                language=language,
                metadata={
                    "pageid": item["pageid"],
                    "source": "wikipedia",
                    "source_lang": lang,
                    "url": f"https://{lang}.wikipedia.org/wiki/{item['title'].replace(' ', '_')}"
                }
            )
            
            articles.append(document)
        
        return articles
    
    def build_corpus(
        self, 
        queries: List[Union[str, Dict[str, Any]]], 
        limit_per_query: int = 5,
        show_progress: bool = True
    ) -> List[Document]:
        """
        Build a corpus by searching Wikipedia for each query
        """
        all_docs = []
        seen_ids = set()
        
        # Extract query texts
        query_texts = []
        for query in queries:
            if isinstance(query, str):
                query_texts.append(query)
            elif isinstance(query, dict):
                # For dictionaries, check common keys
                for key in ["text", "query", "English", "Korean", "EtoK", "KtoE"]:
                    if key in query and isinstance(query[key], str):
                        query_texts.append(query[key])
        
        # Remove duplicates
        query_texts = list(set(query_texts))
        logger.info(f"Building corpus from Wikipedia using {len(query_texts)} unique queries...")
        
        # Search for each query in each language
        progress_iter = tqdm(query_texts, desc="Searching Wikipedia") if show_progress else query_texts
        for query_text in progress_iter:
            for lang in self.langs:
                articles = self.search(
                    query=query_text, 
                    lang=lang, 
                    limit=limit_per_query
                )
                
                # Add to corpus (avoid duplicates)
                for doc in articles:
                    if doc.id not in seen_ids:
                        seen_ids.add(doc.id)
                        all_docs.append(doc)
        
        logger.info(f"Built corpus with {len(all_docs)} documents")
        return all_docs
    
    def save_corpus(self, corpus: List[Document], path: str) -> None:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Convert to dictionaries
        corpus_data = [doc.to_dict() for doc in corpus]
        
        # Save to file
        with open(path, "w", encoding="utf-8") as f:
            json.dump(corpus_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved corpus with {len(corpus)} documents to {path}")
    
    @staticmethod
    def load_corpus(path: str) -> List[Document]:
        with open(path, "r", encoding="utf-8") as f:
            corpus_data = json.load(f)
        
        corpus = [Document.from_dict(item) for item in corpus_data]
        logger.info(f"Loaded corpus with {len(corpus)} documents from {path}")
        
        return corpus