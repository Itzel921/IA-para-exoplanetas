"""
Configuración de MongoDB para Backend de Exoplanetas
NASA Space Apps Challenge 2025

Conexión directa a la base de datos MongoDB especificada.
"""

import logging
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
import pymongo


logger = logging.getLogger(__name__)


class MongoDBConnection:
    """
    Clase para manejar la conexión a MongoDB
    
    Configurada para conectar a:
    URL: http://toiletcrafters.us.to:8081/db/ExoData/datossatelite
    Usuario: manu
    Contraseña: tele123
    """
    
    def __init__(self):
        """Inicializar configuración de MongoDB"""
        # Configuración de conexión específica
        self.host = "toiletcrafters.us.to"
        self.port = 8081
        self.database_name = "ExoData"
        self.collection_name = "datossatelite"
        self.username = "manu"
        self.password = "tele123"
        
        # Variables de conexión
        self.client: Optional[AsyncIOMotorClient] = None
        self.database: Optional[AsyncIOMotorDatabase] = None
        
    async def connect(self) -> bool:
        """Conectar a MongoDB"""
        try:
            # Construir URI de conexión
            mongodb_uri = f"mongodb://{self.username}:{self.password}@{self.host}:{self.port}/{self.database_name}"
            
            # Crear cliente con configuración
            self.client = AsyncIOMotorClient(
                mongodb_uri,
                serverSelectionTimeoutMS=5000,  # 5 segundos timeout
                connectTimeoutMS=10000,         # 10 segundos para conectar
                socketTimeoutMS=30000,          # 30 segundos para operaciones
                maxPoolSize=10                  # Pool de conexiones
            )
            
            # Obtener base de datos
            self.database = self.client[self.database_name]
            
            # Probar conexión
            await self.client.admin.command('ping')
            
            logger.info(f"Conectado exitosamente a MongoDB: {self.host}:{self.port}/{self.database_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error conectando a MongoDB: {e}")
            return False
    
    async def close(self):
        """Cerrar conexión a MongoDB"""
        if self.client:
            self.client.close()
            logger.info("Conexión MongoDB cerrada")
    
    def get_collection(self, collection_name: str = None):
        """Obtener colección específica"""
        if not self.database:
            raise RuntimeError("No hay conexión a la base de datos")
        
        col_name = collection_name or self.collection_name
        return self.database[col_name]
    
    async def test_connection(self) -> bool:
        """Probar conexión a MongoDB"""
        try:
            if self.client:
                await self.client.admin.command('ping')
                return True
            return False
        except Exception as e:
            logger.error(f"Test de conexión falló: {e}")
            return False
    
    async def get_server_info(self) -> dict:
        """Obtener información del servidor MongoDB"""
        try:
            if self.client:
                server_info = await self.client.server_info()
                return {
                    'version': server_info.get('version', 'unknown'),
                    'host': self.host,
                    'port': self.port,
                    'database': self.database_name,
                    'collection': self.collection_name
                }
            return {}
        except Exception as e:
            logger.error(f"Error obteniendo info del servidor: {e}")
            return {}


# Instancia global de conexión
_mongo_connection: Optional[MongoDBConnection] = None


def get_mongo_connection() -> MongoDBConnection:
    """Obtener instancia global de conexión MongoDB"""
    global _mongo_connection
    if _mongo_connection is None:
        _mongo_connection = MongoDBConnection()
    return _mongo_connection


async def init_mongodb() -> MongoDBConnection:
    """Inicializar conexión MongoDB"""
    connection = get_mongo_connection()
    await connection.connect()
    return connection


async def close_mongodb():
    """Cerrar conexión MongoDB"""
    global _mongo_connection
    if _mongo_connection:
        await _mongo_connection.close()
        _mongo_connection = None