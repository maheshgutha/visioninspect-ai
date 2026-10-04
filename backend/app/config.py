"""
Central configuration for VisionInspect AI backend.
Reads settings from environment variables / .env file.
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    MONGO_URI: str = "mongodb://localhost:27017"
    DB_NAME: str = "visioninspect_ai"

    JWT_SECRET: str = "change_this_to_a_long_random_secret_key"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 24 hours

    UPLOAD_DIR: str = "uploads"
    MVTEC_DATASET_PATH: str = "./dataset"

    # PatchCore model (trained by train_patchcore.py)
    PATCHCORE_DIR: str = "model_cache/patchcore"
    PATCHCORE_CORESET_RATIO: float = 0.05
    # Multiplies the trained threshold. 1.0 = as trained; >1 = fewer false
    # alarms (but misses subtle defects); <1 = stricter.
    PATCHCORE_THRESHOLD_SCALE: float = 1.0

    # Product recognition: images that don't match any trained product (or whose
    # PatchCore "defect" covers more than this share of the picture) are treated
    # as "not in the trained setup" and returned as unsupported.
    PATCHCORE_MAX_ANOMALY_AREA: float = 0.5
    IDENTIFIER_CUTOFF_OVERRIDE: float | None = None   # None = use the calibrated value

    CORS_ORIGINS: str = "http://localhost:5173"

    class Config:
        env_file = ".env"
        extra = "ignore"   # an unknown line in .env must never stop the backend from starting

    @property
    def cors_origin_list(self):
        origins = [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]
        return origins


settings = Settings()