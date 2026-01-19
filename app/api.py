from fastapi import FastAPI
from pydantic import BaseModel
from src.recommender import HybridRecommender

app = FastAPI(title="Content Recommendation System API ")

# Load model once at startup
recommender = HybridRecommender()


class RecommendRequest(BaseModel):
    user_id: int
    k: int = 10
    alpha: float = 0.7


@app.get("/")
def health_check():
    return {"status": "API is running"}


@app.post("/recommend")
def recommend_movies(request: RecommendRequest):
    results = recommender.recommend(
        raw_user_id=request.user_id,
        k=request.k,
        alpha=request.alpha
    )

    return {
        "user_id": request.user_id,
        "recommendations": results.to_dict(orient="records")
    }
