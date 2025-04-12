# 위키피디아에서 문서 검색
import requests
import json

def search_wikipedia(query, lang="en", limit=5):
    url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query", 
        "format": "json",
        "list": "search", 
        "srsearch": query,
        "srlimit": limit
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    documents = []
    for item in data.get("query", {}).get("search", []):
        document = {
            "id": f"wiki_{lang}_{item['pageid']}",
            "content": item["snippet"].replace("<span class=\"searchmatch\">", "").replace("</span>", ""),
            "title": item["title"],
            "language": "english" if lang == "en" else "korean"
        }
        documents.append(document)
    
    return documents

# 코드 스위치 쿼리 로드
with open("./code-switch.json", "r", encoding="utf-8") as f:
    queries = json.load(f)

# 각 쿼리에 대해 위키피디아 검색
all_documents = []
for query in queries:
    # 영어 쿼리로 검색
    english_docs = search_wikipedia(query["English"], "en", 5)
    all_documents.extend(english_docs)
    
    # 한국어 쿼리로 검색
    korean_docs = search_wikipedia(query["Korean"], "ko", 5)
    all_documents.extend(korean_docs)

# 중복 제거
unique_docs = {doc["id"]: doc for doc in all_documents}
all_documents = list(unique_docs.values())

# 데이터셋 저장
with open("./documents.json", "w", encoding="utf-8") as f:
    json.dump(all_documents, f, ensure_ascii=False, indent=2)

# 쿼리 데이터셋 생성
query_dataset = []
for idx, query in enumerate(queries):
    for lang_type in ["English", "Korean", "EtoK", "KtoE"]:
        query_dataset.append({
            "id": f"q_{idx}_{lang_type}",
            "content": query[lang_type]
        })

with open("./queries.json", "w", encoding="utf-8") as f:
    json.dump(query_dataset, f, ensure_ascii=False, indent=2)