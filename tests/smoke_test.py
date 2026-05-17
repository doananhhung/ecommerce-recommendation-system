import os
import sys
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.config import config
from src.serving.pipeline import RecommendationPipeline

def test_pipeline_initialization():
    """Kiểm tra xem hệ thống có thể khởi tạo Pipeline và nạp mô hình không."""
    print("Testing Pipeline Initialization...")
    
    if not os.path.exists(config.MODELS_DIR) or not os.path.exists(config.FEATURE_STORE_DIR):
        print("Missing artifacts directories. Please run main_train.py first.")
        sys.exit(1)
        
    pipeline = RecommendationPipeline(models_dir=config.MODELS_DIR, feature_store_dir=config.FEATURE_STORE_DIR)
    
    try:
        pipeline.initialize()
        print("Pipeline initialized successfully.")
    except Exception as e:
        print(f"Pipeline initialization failed: {e}")
        sys.exit(1)
        
    return pipeline

def test_recommendation(pipeline):
    """Kiểm tra luồng predict cho một user có thật trong DB."""
    print("\nTesting Recommendation Generation...")
    
    # Lấy thử 1 user_id từ feature store
    try:
        user_features = pd.read_parquet(config.USER_FEATURES_PATH)
        sample_user_id = user_features['user_id'].iloc[0]
    except Exception as e:
        print(f"Failed to read user features to find a sample user: {e}")
        sys.exit(1)
        
    print(f"Testing inference for User ID: {sample_user_id}")
    
    try:
        result = pipeline.recommend(user_id=int(sample_user_id), top_n=5)
        print("Prediction successful. Response format:")
        print(result)
        
        assert "recommendations" in result, "Missing recommendations key in response"
        assert len(result["recommendations"]) <= 5, "Too many recommendations returned"
        print("Smoke tests passed successfully.")
    except Exception as e:
        print(f"Recommendation test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    pipeline = test_pipeline_initialization()
    test_recommendation(pipeline)
