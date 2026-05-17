from fastapi import FastAPI, HTTPException
import uvicorn

from src.config import config
from src.serving.pipeline import RecommendationPipeline

app = FastAPI(title="Two-Stage Recommender API", version="1.0.0")

# Global pipeline instance
pipeline = None

@app.on_event("startup")
async def startup_event():
    global pipeline
    
    pipeline = RecommendationPipeline(models_dir=config.MODELS_DIR, feature_store_dir=config.FEATURE_STORE_DIR)
    pipeline.initialize()
    print("API Server is ready to receive requests.")

@app.get("/")
def read_root():
    return {"message": "Welcome to Two-Stage Recommender API. Use /recommend/{user_id} to get recommendations."}

@app.get("/recommend/{user_id}")
def get_recommendations(user_id: int, top_n: int = 20):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline is not initialized yet.")
        
    try:
        result = pipeline.recommend(user_id=user_id, top_n=top_n)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommend-by-index/{user_idx}")
def get_recommendations_by_index(user_idx: int, top_n: int = 20):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline is not initialized yet.")

    try:
        result = pipeline.recommend_by_index(user_idx=user_idx, top_n=top_n)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main_serve:app", host="127.0.0.1", port=8000, reload=False)
