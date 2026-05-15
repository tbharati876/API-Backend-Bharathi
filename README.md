# SHL Assessment Recommendation API

## Overview

This project is an AI-powered SHL Assessment Recommendation API built using FastAPI.

The system scrapes SHL assessment catalog data, processes assessment descriptions, and recommends relevant assessments based on user hiring requirements or skill needs.

The application is deployed publicly and provides REST API endpoints with Swagger documentation.

---

## Features

- FastAPI backend
- SHL catalog web scraping
- Intelligent assessment recommendations
- TF-IDF vectorization
- FAISS similarity search
- REST API endpoints
- Swagger UI documentation
- Railway cloud deployment

---

## Tech Stack

- Python
- FastAPI
- BeautifulSoup
- FAISS
- Scikit-learn
- NumPy
- Uvicorn

---

## Installation

Clone the repository:

git clone <github-repo-url>
cd <repo-name>

## Install dependencies:

pip install -r requirements.txt

## Run locally:

python app.py

## API Endpoints

- Home Endpoint
GET /

- Response:

{
  "message": "SHL Recommendation API Running"
}

- Health Check
GET /health

- Response:

{
  "status": "ok"
}
- Chat Recommendation Endpoint
POST /chat

- Request Body:

{
  "messages": [
  {
      "role": "user",
      "content": "I need assessments for software developers with coding and problem solving skills"
    }
  ]
}

- Example Response:

{
  "reply": "Recommended SHL assessments.",
  "recommendations": [
    {
      "name": "SHL Coding Skills Assessment and Simulations",
      "url": "https://www.shl.com/products/assessments/skills-and-simulations/coding-simulations/",
      "test_type": "Assessment"
    }
  ],
  "end_of_conversation": true
}

## Deployment

The application is deployed using Railway.


---

## Project Structure

```bash
project/
│
├── app.py
├── requirements.txt
├── Procfile
└── README.md
