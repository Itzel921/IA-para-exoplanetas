"""
Backend principal para Sistema de Detección de Exoplanetas
NASA Space Apps Challenge 2025

Backend modular enfocado únicamente en lógica interna y conexión a MongoDB.
No incluye APIs, ML ni frontend.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import asyncio

# Configuración de logging básico
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Importar módulos del backend
try:
    from app.database.config import MongoDBConnection
    from app.database.models import ExoplanetData, SatelliteData
    from app.database.services import ExoplanetService
    from app.services.data_processing import DataProcessor
    from app.services.file_processing import FileProcessor
except ImportError as e:
    logger.error(f"Error importing backend modules: {e}")
    sys.exit(1)


class ExoplanetBackend:
    """
    Clase principal del backend de detección de exoplanetas
    
    Maneja:
    - Conexión a MongoDB
    - Procesamiento de datos
    - Operaciones CRUD
    - Lógica de negocio
    """
    
    def __init__(self):
        """Inicializar backend con conexión a MongoDB"""
        self.mongo_connection = None
        self.exoplanet_service = None
        self.data_processor = DataProcessor()
        self.file_processor = FileProcessor()
        
    async def initialize(self):
        """Inicializar conexiones y servicios"""
        try:
            # Conectar a MongoDB
            self.mongo_connection = MongoDBConnection()
            await self.mongo_connection.connect()
            
            # Inicializar servicios
            self.exoplanet_service = ExoplanetService(self.mongo_connection)
            
            logger.info("Backend inicializado correctamente")
            
        except Exception as e:
            logger.error(f"Error inicializando backend: {e}")
            raise
    
    async def close(self):
        """Cerrar conexiones"""
        if self.mongo_connection:
            await self.mongo_connection.close()
            logger.info("Conexiones cerradas")
    
    async def load_satellite_data(self) -> List[Dict]:
        """Cargar datos de satélites desde MongoDB"""
        try:
            data = await self.exoplanet_service.get_all_satellite_data()
            logger.info(f"Cargados {len(data)} registros de satélites")
            return data
        except Exception as e:
            logger.error(f"Error cargando datos de satélites: {e}")
            return []
    
    async def process_exoplanet_data(self, data: Dict) -> Dict:
        """Procesar datos de exoplanetas"""
        try:
            # Procesar con data processor (sin validación)
            processed_data = self.data_processor.process_raw_data(data)
            
            # Guardar en MongoDB
            result = await self.exoplanet_service.save_exoplanet_data(processed_data)
            
            logger.info(f"Datos procesados y guardados: {result}")
            return processed_data
            
        except Exception as e:
            logger.error(f"Error procesando datos: {e}")
            return {}
    
    async def bulk_process_from_file(self, file_path: str) -> Dict:
        """Procesar archivo CSV con datos de exoplanetas"""
        try:
            # Leer archivo
            with open(file_path, 'rb') as f:
                file_content = f.read()
            
            # Procesar archivo
            result = self.file_processor.process_csv_file(file_content, file_path)
            
            if result.get('success'):
                # Procesar cada registro
                processed_records = []
                for record in result.get('data', []):
                    processed = await self.process_exoplanet_data(record)
                    if processed:
                        processed_records.append(processed)
                
                logger.info(f"Procesados {len(processed_records)} registros desde archivo")
                return {
                    'success': True,
                    'processed_count': len(processed_records),
                    'records': processed_records
                }
            
            return {'success': False, 'error': 'Error procesando archivo'}
            
        except Exception as e:
            logger.error(f"Error procesando archivo: {e}")
            return {'success': False, 'error': str(e)}
    
    async def get_database_stats(self) -> Dict:
        """Obtener estadísticas de la base de datos"""
        try:
            stats = await self.exoplanet_service.get_database_statistics()
            logger.info(f"Estadísticas de BD: {stats}")
            return stats
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas: {e}")
            return {}


async def main():
    """Función principal del backend"""
    logger.info("=== Backend Exoplanetas - NASA Space Apps 2025 ===")
    
    # Inicializar backend
    backend = ExoplanetBackend()
    
    try:
        # Inicializar conexiones
        await backend.initialize()
        
        # Cargar datos existentes
        satellite_data = await backend.load_satellite_data()
        logger.info(f"Datos disponibles: {len(satellite_data)} registros")
        
        # Obtener estadísticas
        stats = await backend.get_database_stats()
        
        # Ejemplo de procesamiento de datos
        sample_data = {
            'object_name': 'Test-Exoplanet-001',
            'period': 365.25,
            'radius': 1.0,
            'temperature': 288.0,
            'star_radius': 1.0,
            'star_mass': 1.0,
            'star_temperature': 5778.0,
            'transit_depth': 84.0,
            'transit_duration': 6.5,
            'signal_noise_ratio': 15.0,
            'mission_source': 'Test'
        }
        
        # Procesar datos de prueba
        result = await backend.process_exoplanet_data(sample_data)
        
        logger.info("Backend ejecutado correctamente")
        
    except Exception as e:
        logger.error(f"Error en ejecución principal: {e}")
        
    finally:
        # Cerrar conexiones
        await backend.close()


if __name__ == "__main__":
    # Ejecutar backend
    asyncio.run(main())