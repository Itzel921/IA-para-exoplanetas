"""
FastAPI Backend for Exoplanet Detection System
NASA Space Apps Challenge 2025

Main API application with ensemble ML model serving
Target: 83.08% accuracy using Stacking ensemble
MongoDB integration for data persistence
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import io
import logging
from datetime import datetime
import json
import uuid
import time

# MongoDB integration
from database import (
    database,
    PredictionService,
    BatchAnalysisService,
    UserSessionService,
    PredictionStatus,
    ExoplanetFeatures as DBExoplanetFeatures
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/exoplanet_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Exoplanet Detection API",
    description="AI system for detecting exoplanets using ensemble ML algorithms",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB event handlers
@app.on_event("startup")
async def startup_event():
    """Initialize database connection and load ML models"""
    try:
        # Connect to MongoDB
        await database.connect_to_database()
        logger.info("Database connection established")
        
        # Load ML models
        load_models()
        logger.info("ML models loaded successfully")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

@app.on_event("shutdown") 
async def shutdown_event():
    """Close database connection"""
    await database.close_database_connection()
    logger.info("Application shutdown complete")

# Serve static files (HTML/CSS/JS frontend)
app.mount("/static", StaticFiles(directory="web/frontend"), name="static")

# Global variables for model loading
model = None
preprocessor = None
feature_names = []

# Session management
async def get_or_create_session(session_id: str = None) -> str:
    """Get or create user session"""
    if not session_id:
        session_id = str(uuid.uuid4())
    
    await UserSessionService.create_or_update_session(session_id)
    return session_id

# Pydantic models for API validation
class ExoplanetFeatures(BaseModel):
    """Individual exoplanet candidate parameters"""
    period: float = Field(..., description="Orbital period in days", gt=0)
    radius: float = Field(..., description="Planetary radius in Earth radii", gt=0)
    temp: float = Field(..., description="Equilibrium temperature in Kelvin", gt=0)
    starRadius: float = Field(..., description="Stellar radius in solar radii", gt=0)
    starMass: float = Field(..., description="Stellar mass in solar masses", gt=0)
    starTemp: float = Field(..., description="Stellar temperature in Kelvin", gt=0)
    depth: float = Field(..., description="Transit depth in ppm", gt=0)
    duration: float = Field(..., description="Transit duration in hours", gt=0)
    snr: float = Field(..., description="Signal-to-noise ratio", gt=0)

class PredictionResponse(BaseModel):
    """Single prediction response"""
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    feature_importance: Optional[Dict[str, float]] = None
    analysis_timestamp: str

class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    total_processed: int
    confirmed_planets: int
    candidates: int
    false_positives: int
    results: List[Dict]
    processing_time: float
    analysis_timestamp: str

class ModelInfo(BaseModel):
    """Model information response"""
    model_type: str
    base_models: List[str]
    training_accuracy: float
    features_used: int
    last_updated: str
    target_accuracy: float

def load_models():
    """Load pre-trained ensemble model and preprocessor"""
    global model, preprocessor, feature_names
    
    try:
        # In production, load actual trained models
        # model = joblib.load('models/best_ensemble_model.pkl')
        # preprocessor = joblib.load('models/preprocessor.pkl')
        # feature_names = joblib.load('models/feature_names.pkl')
        
        # For development, create mock objects
        logger.info("Loading mock models for development...")
        model = MockEnsembleModel()
        preprocessor = MockPreprocessor()
        feature_names = [
            'period', 'radius', 'temp', 'starRadius', 'starMass', 
            'starTemp', 'depth', 'duration', 'snr',
            'planet_star_radius_ratio', 'equilibrium_temp_ratio', 
            'transit_depth_expected', 'orbital_velocity'
        ]
        
        logger.info("Models loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise HTTPException(status_code=500, detail="Model loading failed")

class MockEnsembleModel:
    """Mock ensemble model for development"""
    def predict(self, X):
        """Mock prediction"""
        np.random.seed(42)
        return np.random.choice([0, 1], size=len(X), p=[0.7, 0.3])
    
    def predict_proba(self, X):
        """Mock probability prediction"""
        np.random.seed(42)
        probs = np.random.random((len(X), 2))
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs

class MockPreprocessor:
    """Mock preprocessor for development"""
    def transform(self, X):
        """Mock preprocessing"""
        return np.array(X)

def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply astronomical feature engineering
    Based on research achieving 83.08% accuracy
    """
    df_eng = df.copy()
    
    # Physical ratios (key features from research)
    df_eng['planet_star_radius_ratio'] = df['radius'] / df['starRadius']
    df_eng['equilibrium_temp_ratio'] = df['temp'] / df['starTemp']
    
    # Expected transit depth based on physics
    df_eng['transit_depth_expected'] = (df['radius'] / df['starRadius']) ** 2 * 1e6  # ppm
    
    # Orbital characteristics
    df_eng['orbital_velocity'] = 2 * np.pi / df['period']  # Simplified
    
    # Habitable zone distance (simplified)
    hz_distance = df['starTemp'] / 5778.0  # Scaled by Sun's temperature
    df_eng['hz_distance'] = hz_distance
    
    # Signal quality metrics
    df_eng['depth_snr_ratio'] = df['depth'] / df['snr']
    df_eng['duration_period_ratio'] = df['duration'] / (df['period'] * 24)  # Fraction of orbit
    
    logger.info(f"Applied feature engineering. Shape: {df_eng.shape}")
    return df_eng

def get_feature_importance(model, features: Dict) -> Dict[str, float]:
    """Extract feature importance from model"""
    try:
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            return dict(zip(feature_names[:len(importance)], importance.tolist()))
        else:
            # Mock importance for development
            mock_importance = np.random.random(len(feature_names))
            mock_importance = mock_importance / mock_importance.sum()
            return dict(zip(feature_names, mock_importance.tolist()))
    except Exception as e:
        logger.warning(f"Could not extract feature importance: {str(e)}")
        return {}

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    load_models()

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve main HTML page"""
    try:
        with open("web/frontend/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="""
        <html>
            <head><title>Exoplanet Detection System</title></head>
            <body>
                <h1>NASA Space Apps Challenge 2025</h1>
                <h2>Exoplanet Detection with AI</h2>
                <p>Frontend files not found. Please check the web/frontend directory.</p>
                <p><a href="/api/docs">API Documentation</a></p>
            </body>
        </html>
        """)

@app.post("/api/predict", response_model=PredictionResponse)
async def predict_single(features: ExoplanetFeatures, session_id: Optional[str] = None):
    """
    Single exoplanet prediction with MongoDB persistence
    
    Uses ensemble ML model with 83.08% target accuracy
    Stores results in MongoDB for tracking and analytics
    """
    try:
        start_time = time.time()
        
        # Get or create user session
        session_id = await get_or_create_session(session_id)
        
        # Convert to DataFrame
        feature_dict = features.dict()
        df = pd.DataFrame([feature_dict])
        
        logger.info(f"Processing single prediction for period={features.period}")
        
        # Apply feature engineering
        df_engineered = apply_feature_engineering(df)
        
        # Preprocess
        X = preprocessor.transform(df_engineered)
        
        # Prediction
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        
        # Feature importance
        importance = get_feature_importance(model, feature_dict)
        
        # Processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Convert features to DB format
        db_features = DBExoplanetFeatures(
            period=features.period,
            radius=features.radius,
            temp=features.temp,
            star_radius=features.starRadius,
            star_mass=features.starMass,
            star_temp=features.starTemp,
            depth=features.depth,
            duration=features.duration,
            snr=features.snr
        )
        
        # Determine prediction status
        prediction_status = PredictionStatus.CONFIRMED if prediction == 1 else PredictionStatus.FALSE_POSITIVE
        
        # Save to MongoDB
        await PredictionService.save_prediction(
            features=db_features,
            prediction=prediction_status,
            confidence=float(max(probabilities)),
            probabilities={
                "CONFIRMED": float(probabilities[1]),
                "FALSE_POSITIVE": float(probabilities[0])
            },
            processing_time_ms=processing_time_ms,
            user_session=session_id,
            feature_importance=importance
        )
        
        # Update session stats
        await UserSessionService.increment_prediction_count(session_id)
        
        # Prepare response
        result = PredictionResponse(
            prediction="CONFIRMED" if prediction == 1 else "FALSE_POSITIVE",
            confidence=float(max(probabilities)),
            probabilities={
                "CONFIRMED": float(probabilities[1]),
                "FALSE_POSITIVE": float(probabilities[0])
            },
            feature_importance=importance,
            analysis_timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"Prediction completed and saved: {result.prediction} (confidence: {result.confidence:.3f})")
        return result
        
    except Exception as e:
        logger.error(f"Error in single prediction: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/api/batch-predict", response_model=BatchPredictionResponse)
async def batch_predict(file: UploadFile = File(...)):
    """
    Batch prediction from CSV file
    
    Expected columns: period, radius, temp, starRadius, starMass, starTemp, depth, duration, snr
    """
    try:
        start_time = datetime.now()
        
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        logger.info(f"Processing batch file: {file.filename} with {len(df)} rows")
        
        # Validate required columns
        required_columns = ['period', 'radius', 'temp', 'starRadius', 'starMass', 
                          'starTemp', 'depth', 'duration', 'snr']
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {list(missing_columns)}"
            )
        
        # Apply feature engineering
        df_engineered = apply_feature_engineering(df)
        
        # Preprocess
        X = preprocessor.transform(df_engineered)
        
        # Predictions
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        # Compile results
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            result = {
                'index': i,
                'prediction': "CONFIRMED" if pred == 1 else "FALSE_POSITIVE",
                'confidence': float(max(prob)),
                'prob_confirmed': float(prob[1]),
                'prob_false_positive': float(prob[0])
            }
            results.append(result)
        
        # Calculate statistics
        confirmed_count = sum(1 for r in results if r['prediction'] == 'CONFIRMED')
        processing_time = (datetime.now() - start_time).total_seconds()
        
        response = BatchPredictionResponse(
            total_processed=len(results),
            confirmed_planets=confirmed_count,
            candidates=0,  # Could be implemented with threshold logic
            false_positives=len(results) - confirmed_count,
            results=results,
            processing_time=processing_time,
            analysis_timestamp=start_time.isoformat()
        )
        
        logger.info(f"Batch processing completed: {confirmed_count}/{len(results)} confirmed planets")
        return response
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Batch processing error: {str(e)}")

@app.get("/api/model-info", response_model=ModelInfo)
async def get_model_info():
    """Get model information and performance metrics"""
    return ModelInfo(
        model_type="Ensemble (Stacking)",
        base_models=["Random Forest", "AdaBoost", "Extra Trees", "LightGBM"],
        training_accuracy=0.8308,  # Target from research
        features_used=len(feature_names),
        last_updated="2025-01-01",
        target_accuracy=0.83
    )

# New MongoDB-powered endpoints

@app.get("/api/recent-predictions")
async def get_recent_predictions(limit: int = 50, session_id: Optional[str] = None):
    """Get recent predictions from database"""
    try:
        predictions = await PredictionService.get_recent_predictions(limit=limit, user_session=session_id)
        return [
            {
                "id": str(pred.id),
                "prediction": pred.prediction,
                "confidence": pred.confidence,
                "timestamp": pred.timestamp.isoformat(),
                "features": pred.features.dict()
            }
            for pred in predictions
        ]
    except Exception as e:
        logger.error(f"Error getting recent predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/prediction-stats")
async def get_prediction_statistics(days: int = 7):
    """Get prediction statistics for the last N days"""
    try:
        stats = await PredictionService.get_prediction_stats(days=days)
        return stats
    except Exception as e:
        logger.error(f"Error getting prediction stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/batch-analysis/{batch_id}")
async def get_batch_analysis(batch_id: str):
    """Get batch analysis results by ID"""
    try:
        batch = await BatchAnalysisService.get_batch_analysis(batch_id)
        if not batch:
            raise HTTPException(status_code=404, detail="Batch analysis not found")
        
        return {
            "batch_id": batch.batch_id,
            "filename": batch.filename,
            "total_objects": batch.total_objects,
            "confirmed_planets": batch.confirmed_planets,
            "false_positives": batch.false_positives,
            "status": batch.status,
            "processing_start": batch.processing_start.isoformat(),
            "processing_end": batch.processing_end.isoformat() if batch.processing_end else None,
            "average_confidence": batch.average_confidence,
            "high_confidence_detections": batch.high_confidence_detections,
            "results": batch.results
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting batch analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint with database status"""
    try:
        # Test database connection
        stats = await PredictionService.get_prediction_stats(days=1)
        db_healthy = True
    except:
        db_healthy = False
    
    return {
        'status': 'healthy' if db_healthy else 'degraded',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None,
        'database_connected': db_healthy,
        'api_version': '1.0.0'
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)