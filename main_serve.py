from fastapi import FastAPI, HTTPException
import uvicorn
import os

from src.serving.pipeline import RecommendationPipeline

app = FastAPI(title="Two-Stage Recommender API", version="1.0.0")

# Global pipeline instance
pipeline = None

@app.on_event("startup")
async def startup_event():
    global pipeline
    models_dir = os.path.join(os.path.dirname(__file__), 'models_store')
    feature_store_dir = os.path.join(os.path.dirname(__file__), 'data', 'feature_store')
    
    pipeline = RecommendationPipeline(models_dir=models_dir, feature_store_dir=feature_store_dir)
    pipeline.initialize()
    print("API Server is ready to receive requests.")

@app.get("/")
def read_root():
    return {"message": "Welcome to Two-Stage Recommender API. Use /recommend/{user_idx} to get recommendations."}

@app.get("/recommend/{user_idx}")
def get_recommendations(user_idx: int, top_n: int = 20):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline is not initialized yet.")
        
    try:
        result = pipeline.recommend(user_idx=user_idx, top_n=top_n)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main_serve:app", host="127.0.0.1", port=8000, reload=False)
