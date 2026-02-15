import os
import time
import hashlib
import numpy as np
import cohere
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from collections import OrderedDict

# ---------------------------------------------------
# Configuration
# ---------------------------------------------------

MODEL_COST_PER_MILLION = 0.50
AVG_TOKENS_PER_REQUEST = 500
TTL_SECONDS = 86400  # 24 hours
MAX_CACHE_SIZE = 1500
SEMANTIC_THRESHOLD = 0.95

# ---------------------------------------------------
# App Setup
# ---------------------------------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,   # 🔥 change this
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_cohere_client():
    key = os.getenv("CO_API_KEY")
    if not key:
        return None
    return cohere.Client(key)


# ---------------------------------------------------
# Cache + Analytics
# ---------------------------------------------------

cache = OrderedDict()  # LRU behavior
analytics = {
    "totalRequests": 0,
    "cacheHits": 0,
    "cacheMisses": 0
}

# ---------------------------------------------------
# Request Model
# ---------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    application: str


# ---------------------------------------------------
# Utility Functions
# ---------------------------------------------------

def md5_hash(text):
    return hashlib.md5(text.encode()).hexdigest()


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (
        np.linalg.norm(vec1) * np.linalg.norm(vec2)
    )


def remove_expired_entries():
    now = time.time()
    keys_to_delete = [
        key for key, value in cache.items()
        if now - value["created_at"] > TTL_SECONDS
    ]
    for key in keys_to_delete:
        del cache[key]


def evict_if_needed():
    while len(cache) > MAX_CACHE_SIZE:
        cache.popitem(last=False)  # LRU eviction


def generate_answer(query):
    return f"FAQ response: {query}"



def get_embedding(text):
    client = get_cohere_client()
    if not client:
        return None

    try:
        response = client.embed(
            texts=[text],
            model="embed-english-v3.0",
            input_type="search_query"
        )
        return np.array(response.embeddings[0])
    except Exception:
        return None



# ---------------------------------------------------
# Main Endpoint
# ---------------------------------------------------
@app.get("/")
def root():
    return {"status": "alive"}

@app.post("/cache")
def query_endpoint(request: QueryRequest):

    start_time = time.time()
    analytics["totalRequests"] += 1
    remove_expired_entries()

    query = request.query.strip()
    cache_key = md5_hash(query)

    # 1️⃣ Exact Match Check
    if cache_key in cache:
        analytics["cacheHits"] += 1
        cache.move_to_end(cache_key)  # LRU update
        cache[cache_key]["last_used"] = time.time()

        latency = max(1, int((time.time() - start_time) * 1000))

        return {
            "answer": cache[cache_key]["answer"],
            "cached": True,
            "latency": latency,
            "cacheKey": cache_key
        }

    # 2️⃣ Semantic Similarity Check
    query_embedding = get_embedding(query)
    if query_embedding is not None:
        for key, entry in cache.items():
            similarity = cosine_similarity(query_embedding, entry["embedding"])
            if similarity > SEMANTIC_THRESHOLD:
                analytics["cacheHits"] += 1
                cache.move_to_end(key)
                entry["last_used"] = time.time()

                latency = max(1, int((time.time() - start_time) * 1000))

                return {
                    "answer": entry["answer"],
                    "cached": True,

                    "latency": latency,
                    "cacheKey": key
                }

    # 3️⃣ Cache Miss → Call LLM
    analytics["cacheMisses"] += 1

    answer = generate_answer(query)

    cache[cache_key] = {
        "query": query,
        "answer": answer,
        "embedding": query_embedding,
        "created_at": time.time(),
        "last_used": time.time()
    }

    evict_if_needed()

    latency = int((time.time() - start_time) * 1000)

    return {
        "answer": answer,
        "cached": False,
        "latency": latency,
        "cacheKey": cache_key
    }


# ---------------------------------------------------
# Analytics Endpoint
# ---------------------------------------------------

def analytics_endpoint():

    total = analytics["totalRequests"]
    hits = analytics["cacheHits"]
    misses = analytics["cacheMisses"]

    hit_rate = hits / total if total else 0
    baseline_tokens = total * AVG_TOKENS_PER_REQUEST
    actual_tokens = misses * AVG_TOKENS_PER_REQUEST

    savings_tokens = baseline_tokens - actual_tokens
    cost_savings = (savings_tokens / 1_000_000) * MODEL_COST_PER_MILLION
    savings_percent = hit_rate * 100

    return {
        "hitRate": round(hit_rate, 2),
        "totalRequests": total,
        "cacheHits": hits,
        "cacheMisses": misses,
        "cacheSize": len(cache),
        "costSavings": round(cost_savings, 2),
        "savingsPercent": round(savings_percent, 2),
        "strategies": [
            "exact match",
            "semantic similarity",
            "LRU eviction",
            "TTL expiration"
        ]
    }

@app.get("/analytics")
def get_analytics():
    return analytics_endpoint()

@app.get("/cache/analytics")
async def cache_analytics():
    return analytics_endpoint()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))

import time
if not cache:
    time.sleep(1)  # simulate heavy processing
