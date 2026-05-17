import os
from dataclasses import dataclass

@dataclass
class ProjectConfig:
    # Base paths
    ROOT_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR: str = os.path.join(ROOT_DIR, 'data')
    MODELS_DIR: str = os.path.join(ROOT_DIR, 'models_store')
    
    # Specific paths
    RAW_DATA_PATH: str = os.path.join(DATA_DIR, 'raw', '2019-Oct.csv')
    SESSIONS_DIR: str = os.path.join(DATA_DIR, 'sessions')
    FEATURE_STORE_DIR: str = os.path.join(DATA_DIR, 'feature_store')
    
    LABELED_SESSIONS_PATH: str = os.path.join(SESSIONS_DIR, 'labeled_sessions.parquet')
    USER_FEATURES_PATH: str = os.path.join(FEATURE_STORE_DIR, 'user_features.parquet')
    ITEM_FEATURES_PATH: str = os.path.join(FEATURE_STORE_DIR, 'item_features.parquet')
    ITEM_EMBEDDINGS_PATH: str = os.path.join(FEATURE_STORE_DIR, 'item_embeddings.npy')
    
    RECALL_WEIGHTS_PATH: str = os.path.join(MODELS_DIR, 'recall_weights.pth')
    RANKER_MODEL_PATH: str = os.path.join(MODELS_DIR, 'ranker_model.txt')
    USER_ENCODER_PATH: str = os.path.join(MODELS_DIR, 'user_encoder.joblib')
    ITEM_ENCODER_PATH: str = os.path.join(MODELS_DIR, 'item_encoder.joblib')
    RECALL_METADATA_PATH: str = os.path.join(MODELS_DIR, 'recall_metadata.json')

    # Data Pipeline Hyperparameters
    PIPELINE_NROWS: int = 1000000  # Number of rows to read from raw data. Set to None for all.
    SESSION_THRESHOLD_MINUTES: int = 30
    
    # Recall Model Hyperparameters
    RECALL_BATCH_SIZE: int = 4096
    RECALL_EMBEDDING_DIM: int = 64
    RECALL_LEARNING_RATE: float = 0.01
    RECALL_EPOCHS: int = 5
    RECALL_NEG_SAMPLE_RATIO: int = 4 # 1 positive : X negatives

    # Ranking Model Hyperparameters
    RANKING_LEARNING_RATE: float = 0.05
    RANKING_NUM_LEAVES: int = 31
    RANKING_NUM_BOOST_ROUND: int = 100
    RANKING_TEST_SIZE: float = 0.2
    
    # Serving Hyperparameters
    SERVING_TOP_K_RECALL: int = 200
    SERVING_TOP_N_RANKING: int = 20

config = ProjectConfig()
