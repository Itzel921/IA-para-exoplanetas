"""
Database Services for Exoplanet Detection System
NASA Space Apps Challenge 2025

Async service layer for MongoDB operations
Implements patterns from web-interface-deployment.md
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from .models import (
    PredictionResult, 
    BatchAnalysis, 
    ModelMetrics, 
    UserSession,
    ExoplanetFeatures,
    PredictionStatus
)
import uuid
import logging

logger = logging.getLogger(__name__)

class PredictionService:
    """Service for managing prediction results"""
    
    @staticmethod
    async def save_prediction(
        features: ExoplanetFeatures,
        prediction: PredictionStatus,
        confidence: float,
        probabilities: Dict[str, float],
        processing_time_ms: float = None,
        user_session: str = None,
        feature_importance: Dict[str, float] = None
    ) -> PredictionResult:
        """Save a single prediction result"""
        
        prediction_doc = PredictionResult(
            features=features,
            prediction=prediction,
            confidence=confidence,
            probabilities=probabilities,
            processing_time_ms=processing_time_ms,
            user_session=user_session,
            feature_importance=feature_importance
        )
        
        await prediction_doc.save()
        logger.info(f"Saved prediction: {prediction} (confidence: {confidence:.3f})")
        return prediction_doc
    
    @staticmethod
    async def get_recent_predictions(
        limit: int = 50,
        user_session: str = None
    ) -> List[PredictionResult]:
        """Get recent predictions, optionally filtered by user"""
        
        query = {}
        if user_session:
            query["user_session"] = user_session
            
        predictions = await PredictionResult.find(query) \
            .sort([("timestamp", -1)]) \
            .limit(limit) \
            .to_list()
        
        return predictions
    
    @staticmethod
    async def get_prediction_stats(
        days: int = 7
    ) -> Dict[str, Any]:
        """Get prediction statistics for the last N days"""
        
        start_date = datetime.utcnow() - timedelta(days=days)
        
        # Aggregate statistics
        pipeline = [
            {"$match": {"timestamp": {"$gte": start_date}}},
            {"$group": {
                "_id": "$prediction",
                "count": {"$sum": 1},
                "avg_confidence": {"$avg": "$confidence"}
            }}
        ]
        
        results = await PredictionResult.aggregate(pipeline).to_list()
        
        stats = {
            "total_predictions": sum(r["count"] for r in results),
            "confirmed_planets": 0,
            "false_positives": 0,
            "avg_confidence_confirmed": 0,
            "avg_confidence_false_positive": 0
        }
        
        for result in results:
            if result["_id"] == PredictionStatus.CONFIRMED:
                stats["confirmed_planets"] = result["count"]
                stats["avg_confidence_confirmed"] = result["avg_confidence"]
            elif result["_id"] == PredictionStatus.FALSE_POSITIVE:
                stats["false_positives"] = result["count"]
                stats["avg_confidence_false_positive"] = result["avg_confidence"]
        
        return stats

class BatchAnalysisService:
    """Service for managing batch analyses"""
    
    @staticmethod
    async def create_batch_analysis(
        filename: str,
        total_objects: int
    ) -> BatchAnalysis:
        """Create a new batch analysis record"""
        
        batch_analysis = BatchAnalysis(
            batch_id=str(uuid.uuid4()),
            filename=filename,
            total_objects=total_objects,
            processing_start=datetime.utcnow(),
            status="processing"
        )
        
        await batch_analysis.save()
        logger.info(f"Created batch analysis: {batch_analysis.batch_id}")
        return batch_analysis
    
    @staticmethod
    async def update_batch_progress(
        batch_id: str,
        results: List[Dict],
        status: str = "processing"
    ) -> Optional[BatchAnalysis]:
        """Update batch analysis with new results"""
        
        batch = await BatchAnalysis.find_one({"batch_id": batch_id})
        if not batch:
            return None
        
        batch.results.extend(results)
        batch.status = status
        
        # Update counters
        for result in results:
            if result["prediction"] == PredictionStatus.CONFIRMED:
                batch.confirmed_planets += 1
            elif result["prediction"] == PredictionStatus.FALSE_POSITIVE:
                batch.false_positives += 1
            else:
                batch.candidates += 1
        
        # Calculate statistics
        if batch.results:
            confidences = [r["confidence"] for r in batch.results]
            batch.average_confidence = sum(confidences) / len(confidences)
            batch.high_confidence_detections = sum(
                1 for c in confidences if c > 0.8
            )
        
        if status == "completed":
            batch.processing_end = datetime.utcnow()
        
        await batch.save()
        return batch
    
    @staticmethod
    async def get_batch_analysis(batch_id: str) -> Optional[BatchAnalysis]:
        """Get batch analysis by ID"""
        return await BatchAnalysis.find_one({"batch_id": batch_id})

class ModelMetricsService:
    """Service for managing model performance metrics"""
    
    @staticmethod
    async def save_model_metrics(
        accuracy: float,
        precision: float,
        recall: float,
        f1_score: float,
        roc_auc: float,
        completeness: float,
        reliability: float,
        false_discovery_rate: float,
        test_set_size: int,
        positive_class_proportion: float,
        hyperparameters: Dict = None,
        feature_count: int = None
    ) -> ModelMetrics:
        """Save model evaluation metrics"""
        
        metrics = ModelMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            roc_auc=roc_auc,
            completeness=completeness,
            reliability=reliability,
            false_discovery_rate=false_discovery_rate,
            test_set_size=test_set_size,
            positive_class_proportion=positive_class_proportion,
            hyperparameters=hyperparameters,
            feature_count=feature_count
        )
        
        await metrics.save()
        logger.info(f"Saved model metrics: accuracy={accuracy:.4f}")
        return metrics
    
    @staticmethod
    async def get_latest_metrics() -> Optional[ModelMetrics]:
        """Get the most recent model metrics"""
        return await ModelMetrics.find_one(
            sort=[("evaluation_date", -1)]
        )
    
    @staticmethod
    async def get_metrics_history(days: int = 30) -> List[ModelMetrics]:
        """Get model metrics history"""
        start_date = datetime.utcnow() - timedelta(days=days)
        
        return await ModelMetrics.find(
            {"evaluation_date": {"$gte": start_date}}
        ).sort([("evaluation_date", -1)]).to_list()

class UserSessionService:
    """Service for managing user sessions"""
    
    @staticmethod
    async def create_or_update_session(session_id: str) -> UserSession:
        """Create new session or update existing one"""
        
        session = await UserSession.find_one({"session_id": session_id})
        
        if session:
            session.last_activity = datetime.utcnow()
            await session.save()
        else:
            session = UserSession(session_id=session_id)
            await session.save()
            logger.info(f"Created new user session: {session_id}")
        
        return session
    
    @staticmethod
    async def increment_prediction_count(session_id: str) -> Optional[UserSession]:
        """Increment prediction count for session"""
        
        session = await UserSession.find_one({"session_id": session_id})
        if session:
            session.predictions_count += 1
            session.last_activity = datetime.utcnow()
            await session.save()
        
        return session