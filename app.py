import requests
from bs4 import BeautifulSoup
import json
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

import uvicorn


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="SHL Assessment Recommendation API",
    description="AI-powered SHL assessment recommendation system",
    version="1.0.0"
)


# =========================================================
# SCRAPE SHL CATALOG
# =========================================================

CATALOG_URL = "https://www.shl.com/solutions/products/product-catalog/"

headers = {
    "User-Agent": "Mozilla/5.0"
}

print("Scraping SHL catalog...")

response = requests.get(CATALOG_URL, headers=headers)

soup = BeautifulSoup(response.text, "lxml")

links = soup.find_all("a")

assessment_links = []

for link in links:

    href = link.get("href")

    if href and "/products/" in href:

        if href.startswith("/"):
            full_url = "https://www.shl.com" + href
        else:
            full_url = href

        if full_url not in assessment_links:
            assessment_links.append(full_url)

print("Assessment links found:", len(assessment_links))


# =========================================================
# SCRAPE ALL ASSESSMENTS
# =========================================================

all_assessments = []

for url in assessment_links:

    try:

        response = requests.get(url, headers=headers, timeout=20)

        soup = BeautifulSoup(response.text, "lxml")

        title = soup.find("h1")

        paragraphs = soup.find_all("p")

        description = ""

        for p in paragraphs:

            text = p.text.strip()

            if len(text) > 50:
                description = text
                break

        assessment = {
            "name": title.text.strip() if title else "",
            "url": url,
            "description": description
        }

        if assessment["name"] != "":

            all_assessments.append(assessment)

            print("Done:", assessment["name"])

    except Exception as e:

        print("Error:", url)
        print(str(e))

print("Total assessments:", len(all_assessments))


# =========================================================
# SAVE DATASET
# =========================================================

with open("catalog.json", "w") as f:
    json.dump(all_assessments, f, indent=4)

print("catalog.json saved")


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded")


# =========================================================
# PREPARE DOCUMENTS
# =========================================================

documents = []

for item in all_assessments:

    text = item["name"] + " " + item["description"]

    documents.append(text)


# =========================================================
# CREATE EMBEDDINGS
# =========================================================

print("Creating embeddings...")

embeddings = model.encode(documents)

print("Embeddings created")


# =========================================================
# CREATE FAISS INDEX
# =========================================================

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings))

print("FAISS index created")


# =========================================================
# REQUEST SCHEMA
# =========================================================

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]


# =========================================================
# HEALTH ENDPOINT
# =========================================================

@app.get("/")
def home():

    return {
        "message": "SHL Recommendation API Running"
    }


@app.get("/health")
def health():

    return {
        "status": "ok"
    }


# =========================================================
# RECOMMENDATION FUNCTION
# =========================================================

def get_recommendations(user_query, top_k=5):

    query_embedding = model.encode([user_query])

    distances, indices = index.search(
        np.array(query_embedding),
        top_k
    )

    recommendations = []

    for i in indices[0]:

        item = all_assessments[i]

        recommendations.append({
            "name": item["name"],
            "url": item["url"],
            "test_type": "Assessment"
        })

    return recommendations


# =========================================================
# VAGUE QUERY DETECTION
# =========================================================

def is_vague_query(text):

    vague_words = [
        "assessment",
        "test",
        "hiring",
        "job",
        "candidate"
    ]

    text = text.lower()

    if len(text.split()) <= 3:
        return True

    for word in vague_words:

        if text.strip() == word:
            return True

    return False


# =========================================================
# COMPARISON FUNCTION
# =========================================================

def compare_assessments(conversation_text):

    comparison_words = [
        "difference",
        "compare",
        "vs",
        "versus"
    ]

    found = False

    for word in comparison_words:

        if word in conversation_text.lower():
            found = True

    if not found:
        return None

    matched = []

    for item in all_assessments:

        if item["name"].lower() in conversation_text.lower():

            matched.append(item)

    if len(matched) < 2:
        return None

    reply = ""

    for item in matched[:2]:

        reply += (
            f"{item['name']}: "
            f"{item['description']}\n\n"
        )

    return {
        "reply": reply,
        "recommendations": [],
        "end_of_conversation": False
    }


# =========================================================
# CHAT ENDPOINT
# =========================================================

@app.post("/chat")
def chat(request: ChatRequest):

    messages = request.messages

    conversation_text = ""

    for message in messages:

        conversation_text += message.content + " "

    # =====================================================
    # OFF TOPIC REFUSAL
    # =====================================================

    off_topic_words = [
        "law",
        "legal",
        "salary",
        "politics",
        "weather"
    ]

    for word in off_topic_words:

        if word in conversation_text.lower():

            return {
                "reply": "I can only help with SHL assessment recommendations.",
                "recommendations": [],
                "end_of_conversation": False
            }

    # =====================================================
    # COMPARISON SUPPORT
    # =====================================================

    comparison_result = compare_assessments(
        conversation_text
    )

    if comparison_result:

        return comparison_result

    # =====================================================
    # CLARIFICATION
    # =====================================================

    if is_vague_query(conversation_text):

        return {
            "reply": "Could you specify the role, skills, seniority level, or assessment needs?",
            "recommendations": [],
            "end_of_conversation": False
        }

    # =====================================================
    # PERSONALITY TEST SUPPORT
    # =====================================================

    personality_words = [
        "personality",
        "behavior",
        "leadership"
    ]

    personality_needed = False

    for word in personality_words:

        if word in conversation_text.lower():

            personality_needed = True

    # =====================================================
    # GET RECOMMENDATIONS
    # =====================================================

    recommendations = get_recommendations(
        conversation_text,
        top_k=10
    )

    # =====================================================
    # FILTER EMPTY ITEMS
    # =====================================================

    cleaned = []

    for item in recommendations:

        if item["name"] != "":

            cleaned.append(item)

    recommendations = cleaned[:10]

    # =====================================================
    # REPLY
    # =====================================================

    reply = "Here are recommended SHL assessments."

    if personality_needed:

        reply += " Personality-related assessments were also considered."

    # =====================================================
    # FINAL RESPONSE
    # =====================================================

    return {
        "reply": reply,
        "recommendations": recommendations,
        "end_of_conversation": True
    }


# =========================================================
# START SERVER
# =========================================================

import os

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
