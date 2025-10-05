"""
Servicios de base de datos - Backend Exoplanetas
NASA Space Apps Challenge 2025

Servicios simplificados para operaciones CRUD con MongoDB.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from motor.motor_asyncio import AsyncIOMotorCollection
from pymongo.errors import PyMongoError

from .config import MongoDBConnection
from .models import ExoplanetData, SatelliteData, ProcessingResult, DatabaseHelper


class ExoplanetService:
    """
    Servicio para operaciones con datos de exoplanetas
    """
    
    def __init__(self):
        self.db_connection = MongoDBConnection()
        self.collection: Optional[AsyncIOMotorCollection] = None
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self):
        """Inicializar conexión a MongoDB"""
        try:
            await self.db_connection.connect()
            database = await self.db_connection.get_database()
            self.collection = database.exoplanets
            self.logger.info("ExoplanetService inicializado correctamente")
        except Exception as e:
            self.logger.error(f"Error inicializando ExoplanetService: {str(e)}")
            raise
    
    async def insert_exoplanet(self, exoplanet_data: ExoplanetData) -> str:
        """
        Insertar un exoplaneta en la base de datos
        
        Args:
            exoplanet_data: Datos del exoplaneta
        
        Returns:
            ID del documento insertado
        """
        try:
            data_dict = exoplanet_data.to_dict()
            clean_data = DatabaseHelper.prepare_for_mongodb(data_dict)
            
            result = await self.collection.insert_one(clean_data)
            self.logger.info(f"Exoplaneta insertado: {exoplanet_data.object_name}")
            return str(result.inserted_id)
            
        except PyMongoError as e:
            self.logger.error(f"Error insertando exoplaneta: {str(e)}")
            raise
    
    async def bulk_insert_exoplanets(self, exoplanets: List[ExoplanetData]) -> List[str]:
        """
        Insertar múltiples exoplanetas
        
        Args:
            exoplanets: Lista de datos de exoplanetas
        
        Returns:
            Lista de IDs insertados
        """
        try:
            documents = []
            for exoplanet in exoplanets:
                data_dict = exoplanet.to_dict()
                clean_data = DatabaseHelper.prepare_for_mongodb(data_dict)
                documents.append(clean_data)
            
            result = await self.collection.insert_many(documents)
            self.logger.info(f"Insertados {len(documents)} exoplanetas")
            return [str(oid) for oid in result.inserted_ids]
            
        except PyMongoError as e:
            self.logger.error(f"Error en inserción masiva: {str(e)}")
            raise
    
    async def find_exoplanet_by_name(self, object_name: str) -> Optional[ExoplanetData]:
        """
        Buscar exoplaneta por nombre
        
        Args:
            object_name: Nombre del objeto
        
        Returns:
            Datos del exoplaneta o None si no se encuentra
        """
        try:
            document = await self.collection.find_one({"object_name": object_name})
            if document:
                clean_doc = DatabaseHelper.clean_mongodb_result(document)
                return ExoplanetData.from_dict(clean_doc)
            return None
            
        except PyMongoError as e:
            self.logger.error(f"Error buscando exoplaneta: {str(e)}")
            raise
    
    async def find_exoplanets_by_mission(self, mission: str) -> List[ExoplanetData]:
        """
        Buscar exoplanetas por misión
        
        Args:
            mission: Nombre de la misión (Kepler, TESS, K2)
        
        Returns:
            Lista de exoplanetas de la misión
        """
        try:
            cursor = self.collection.find({"mission_source": mission})
            exoplanets = []
            
            async for document in cursor:
                clean_doc = DatabaseHelper.clean_mongodb_result(document)
                exoplanets.append(ExoplanetData.from_dict(clean_doc))
            
            self.logger.info(f"Encontrados {len(exoplanets)} exoplanetas de {mission}")
            return exoplanets
            
        except PyMongoError as e:
            self.logger.error(f"Error buscando por misión: {str(e)}")
            raise
    
    async def count_exoplanets(self) -> int:
        """Contar total de exoplanetas"""
        try:
            count = await self.collection.count_documents({})
            return count
        except PyMongoError as e:
            self.logger.error(f"Error contando exoplanetas: {str(e)}")
            return 0


class SatelliteService:
    """
    Servicio para operaciones con datos de satélites
    """
    
    def __init__(self):
        self.db_connection = MongoDBConnection()
        self.collection: Optional[AsyncIOMotorCollection] = None
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self):
        """Inicializar conexión a MongoDB"""
        try:
            await self.db_connection.connect()
            database = await self.db_connection.get_database()
            self.collection = database.satellites
            self.logger.info("SatelliteService inicializado correctamente")
        except Exception as e:
            self.logger.error(f"Error inicializando SatelliteService: {str(e)}")
            raise
    
    async def insert_satellite(self, satellite_data: SatelliteData) -> str:
        """
        Insertar datos de satélite
        
        Args:
            satellite_data: Datos del satélite
        
        Returns:
            ID del documento insertado
        """
        try:
            data_dict = satellite_data.to_dict()
            clean_data = DatabaseHelper.prepare_for_mongodb(data_dict)
            
            result = await self.collection.insert_one(clean_data)
            self.logger.info(f"Satélite insertado: {satellite_data.name}")
            return str(result.inserted_id)
            
        except PyMongoError as e:
            self.logger.error(f"Error insertando satélite: {str(e)}")
            raise
    
    async def find_satellite_by_id(self, satellite_id: str) -> Optional[SatelliteData]:
        """
        Buscar satélite por ID
        
        Args:
            satellite_id: ID del satélite
        
        Returns:
            Datos del satélite o None si no se encuentra
        """
        try:
            document = await self.collection.find_one({"satellite_id": satellite_id})
            if document:
                clean_doc = DatabaseHelper.clean_mongodb_result(document)
                return SatelliteData.from_dict(clean_doc)
            return None
            
        except PyMongoError as e:
            self.logger.error(f"Error buscando satélite: {str(e)}")
            raise


class ProcessingService:
    """
    Servicio para registrar resultados de procesamiento
    """
    
    def __init__(self):
        self.db_connection = MongoDBConnection()
        self.collection: Optional[AsyncIOMotorCollection] = None
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self):
        """Inicializar conexión a MongoDB"""
        try:
            await self.db_connection.connect()
            database = await self.db_connection.get_database()
            self.collection = database.processing_results
            self.logger.info("ProcessingService inicializado correctamente")
        except Exception as e:
            self.logger.error(f"Error inicializando ProcessingService: {str(e)}")
            raise
    
    async def log_processing_result(self, result: ProcessingResult) -> str:
        """
        Registrar resultado de procesamiento
        
        Args:
            result: Resultado del procesamiento
        
        Returns:
            ID del documento insertado
        """
        try:
            data_dict = result.to_dict()
            clean_data = DatabaseHelper.prepare_for_mongodb(data_dict)
            
            db_result = await self.collection.insert_one(clean_data)
            self.logger.info(f"Resultado de procesamiento registrado: {result.processing_id}")
            return str(db_result.inserted_id)
            
        except PyMongoError as e:
            self.logger.error(f"Error registrando resultado: {str(e)}")
            raise


class DataManagerService:
    """
    Servicio principal que coordina todos los servicios de datos
    """
    
    def __init__(self):
        self.exoplanet_service = ExoplanetService()
        self.satellite_service = SatelliteService()
        self.processing_service = ProcessingService()
        self.logger = logging.getLogger(__name__)
    
    async def initialize_all_services(self):
        """Inicializar todos los servicios"""
        try:
            await self.exoplanet_service.initialize()
            await self.satellite_service.initialize()
            await self.processing_service.initialize()
            self.logger.info("Todos los servicios inicializados correctamente")
        except Exception as e:
            self.logger.error(f"Error inicializando servicios: {str(e)}")
            raise
    
    async def get_database_status(self) -> Dict[str, Any]:
        """
        Obtener estado de la base de datos
        
        Returns:
            Diccionario con estadísticas de la BD
        """
        try:
            exoplanet_count = await self.exoplanet_service.count_exoplanets()
            
            status = {
                "database_connected": True,
                "exoplanet_count": exoplanet_count,
                "last_check": datetime.now().isoformat(),
                "status": "healthy"
            }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error obteniendo estado de BD: {str(e)}")
            return {
                "database_connected": False,
                "error": str(e),
                "last_check": datetime.now().isoformat(),
                "status": "error"
            }