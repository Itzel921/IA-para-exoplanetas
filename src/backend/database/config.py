"""
MongoDB Configuration for Exoplanet Detection System
NASA Space Apps Challenge 2025

Database configuration following the architecture from:
.github/context/web-interface-deployment.md
"""

import os
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class DatabaseConfig:
    """MongoDB configuration and connection management"""
    
    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.database_name = os.getenv("MONGODB_DATABASE", "exoplanets_db")
        self.connection_string = os.getenv(
            "MONGODB_URL", 
            "mongodb://localhost:27017"
        )
    
    async def connect_to_database(self):
        """Establish connection to MongoDB"""
        try:
            self.client = AsyncIOMotorClient(self.connection_string)
            
            # Test connection
            await self.client.admin.command('ping')
            logger.info(f"Connected to MongoDB at {self.connection_string}")
            
            # Initialize Beanie with document models
            from .models import PredictionResult, BatchAnalysis, ModelMetrics
            
            await init_beanie(
                database=self.client[self.database_name],
                document_models=[
                    PredictionResult,
                    BatchAnalysis, 
                    ModelMetrics
                ]
            )
            
            logger.info("Beanie ODM initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    async def close_database_connection(self):
        """Close database connection"""
        if self.client:
            self.client.close()
            logger.info("Disconnected from MongoDB")

# Global database instance
database = DatabaseConfig()