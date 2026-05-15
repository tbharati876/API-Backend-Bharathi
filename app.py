import os
import requests
import json
import numpy as np
import faiss

from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

import uvicorn


app = FastAPI(
    title="SHL Assessment Recommendation API"
)


# =====================================================
# GLOBAL VARIABLES
# =====================================================

model = None
index = None
all_assessments = []


# =====================================================
# STARTUP EVENT
# =====================================================

@app.on_event("startup")
def startup_event():

    global model
    global index
    global all_assessments

    print("Starting initialization...")

    url = "https://www.shl.com/solutions/products/product-catalog/"

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(url, headers=headers)

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

    for item_url in assessment_links[:50]:

        try:

            r = requests.get(item_url, headers=headers, timeout=10)

            s = BeautifulSoup(r.text, "lxml")

            title = s.find("h1")

            paragraphs = s.find_all("p")

            description = ""

            for p in paragraphs:

                text = p.text.strip()

                if len(text) > 50:
                    description = text
                    break

            assessment = {
                "name": title.text.strip() if title else "",
                "url": item_url,
                "description": description
            }

            if assessment["name"] != "":
                all_assessments.append(assessment)

        except:
            pass

    print("Assessments loaded:", len(all_assessments))

    model = SentenceTransformer("all-MiniLM-L6-v2")

    documents = []

    for item in all_assessments:

        text = item["name"] + " " + item["description"]

        documents.append(text)

    embeddings = model.encode(documents)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(np.array(embeddings))

    print("FAISS ready")


# =====================================================
# REQUEST SCHEMA
# =====================================================

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]


# =====================================================
# HOME
# =====================================================

@app.get("/")
def home():

    return {
        "message": "API Running"
    }


@app.get("/health")
def health():

    return {
        "status": "ok"
    }


# =====================================================
# RECOMMENDATIONS
# =====================================================

def get_recommendations(query, top_k=5):

    query_embedding = model.encode([query])

    distances, indices = index.search(
        np.array(query_embedding),
        top_k
    )

    recommendations = []

    for i in indices[0]:

        item = all_assessments[i]

        recommendations.append({
            "name": item["name"],
            "url": item["url"]
        })

    return recommendations


# =====================================================
# CHAT
# =====================================================

@app.post("/chat")
def chat(request: ChatRequest):

    conversation = ""

    for msg in request.messages:

        conversation += msg.content + " "

    recommendations = get_recommendations(
        conversation,
        top_k=5
    )

    return {
        "reply": "Recommended assessments",
        "recommendations": recommendations
    }


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
