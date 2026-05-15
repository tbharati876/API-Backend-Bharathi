import os
import requests
import numpy as np
import faiss

from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

import uvicorn

# FASTAPI APP

app = FastAPI(
    title="SHL Assessment Recommendation API",
    version="1.0.0"
)

all_assessments = []
vectorizer = None
index = None

# STARTUP

@app.on_event("startup")
def startup_event():
    
    global all_assessments
    global vectorizer
    global index

    print("Initializing app...")

    url = "https://www.shl.com/solutions/products/product-catalog/"

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(url, headers=headers)

    soup = BeautifulSoup(response.text, "lxml")

    links = soup.find_all("a")

    assessment_links = []

    # COLLECT ASSESSMENT LINKS

    for link in links:

        href = link.get("href")

        if href:

            if "product-catalog/?start=" in href:
                continue

            if "/products/" in href:

                if href.startswith("/"):
                    full_url = "https://www.shl.com" + href
                else:
                    full_url = href

                if "product-catalog" in full_url:
                    continue

                if full_url not in assessment_links:
                    assessment_links.append(full_url)

    print("Valid links found:", len(assessment_links))

    # SCRAPE ASSESSMENTS

    bad_titles = [
        "Find assessments that best meet your needs."
    ]

    for item_url in assessment_links[:40]:

        try:

            r = requests.get(
                item_url,
                headers=headers,
                timeout=10
            )

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

            if (
                assessment["name"] != ""
                and assessment["description"] != ""
                and assessment["name"] not in bad_titles
            ):

                all_assessments.append(assessment)

                print("Loaded:", assessment["name"])

        except Exception as e:

            print("Error:", item_url)

    print("Final assessments:", len(all_assessments))

    # TF-IDF VECTORIZER

    documents = []

    for item in all_assessments:

        text = item["name"] + " " + item["description"]

        documents.append(text)

    vectorizer = TfidfVectorizer()

    embeddings = vectorizer.fit_transform(
        documents
    ).toarray()

    embeddings = np.array(
        embeddings
    ).astype("float32")

    # FAISS INDEX

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    print("FAISS ready")

# REQUEST MODELS

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]

# HOME ENDPOINT

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

# RECOMMENDATION FUNCTION

def get_recommendations(query, top_k=5):

    query_embedding = vectorizer.transform(
        [query]
    ).toarray()

    query_embedding = np.array(
        query_embedding
    ).astype("float32")

    distances, indices = index.search(
        query_embedding,
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

# CHAT ENDPOINT

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
        "reply": "Recommended SHL assessments.",
        "recommendations": recommendations,
        "end_of_conversation": True
    }

# MAIN

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 8080))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
