#!/usr/bin/env python3
"""
Script para poblar MongoDB con datos de prueba para el sistema de detección de exoplanetas
NASA Space Apps Challenge 2025

Este script crea datos sintéticos realistas basados en los datasets KOI, TOI y K2
para testing del frontend y backend.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any
import random
import math
import numpy as np
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie
import sys
import os

# Agregar el directorio del backend al path
backend_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'backend')
sys.path.insert(0, backend_path)

from app.database.models import ExoplanetData, SatelliteData
from app.core.config import settings

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ExoplanetDataGenerator:
    """Generador de datos sintéticos realistas de exoplanetas"""
    
    def __init__(self):
        self.planet_types = {
            'earth_like': {'radius_range': (0.8, 1.2), 'period_range': (200, 500)},
            'super_earth': {'radius_range': (1.2, 2.0), 'period_range': (10, 200)},
            'mini_neptune': {'radius_range': (2.0, 4.0), 'period_range': (5, 100)},
            'hot_jupiter': {'radius_range': (8.0, 15.0), 'period_range': (1, 10)},
            'cold_jupiter': {'radius_range': (10.0, 20.0), 'period_range': (100, 2000)}
        }
        
        self.stellar_types = {
            'M_dwarf': {'temp_range': (2300, 3800), 'radius_range': (0.1, 0.6), 'mass_range': (0.08, 0.6)},
            'K_dwarf': {'temp_range': (3700, 5200), 'radius_range': (0.6, 0.9), 'mass_range': (0.6, 0.9)},
            'G_dwarf': {'temp_range': (5200, 6000), 'radius_range': (0.8, 1.2), 'mass_range': (0.8, 1.2)},
            'F_dwarf': {'temp_range': (6000, 7500), 'radius_range': (1.0, 1.5), 'mass_range': (1.0, 1.5)},
            'A_dwarf': {'temp_range': (7500, 10000), 'radius_range': (1.4, 2.5), 'mass_range': (1.4, 2.5)}
        }
    
    def generate_stellar_parameters(self, stellar_type: str = None) -> Dict[str, float]:
        """Genera parámetros estelares realistas"""
        if not stellar_type:
            stellar_type = random.choice(list(self.stellar_types.keys()))
        
        params = self.stellar_types[stellar_type]
        
        star_temp = random.uniform(*params['temp_range'])
        star_radius = random.uniform(*params['radius_range'])
        star_mass = random.uniform(*params['mass_range'])
        
        # Log g basado en masa y radio (aproximación)
        star_logg = math.log10(star_mass / (star_radius ** 2)) + 4.44  # Solar values
        
        return {
            'star_temp': star_temp,
            'star_radius': star_radius, 
            'star_mass': star_mass,
            'star_logg': star_logg,
            'stellar_type': stellar_type
        }
    
    def generate_planetary_parameters(self, planet_type: str = None, star_params: Dict = None) -> Dict[str, float]:
        """Genera parámetros planetarios realistas"""
        if not planet_type:
            planet_type = random.choice(list(self.planet_types.keys()))
        
        params = self.planet_types[planet_type]
        
        # Parámetros básicos
        radius = random.uniform(*params['radius_range'])
        period = random.uniform(*params['period_range'])
        
        # Temperatura de equilibrio (depende de la estrella)
        if star_params:
            # Usar ley de Stefan-Boltzmann simplificada
            star_temp = star_params.get('star_temp', 5778)
            star_radius = star_params.get('star_radius', 1.0)
            # Distancia orbital aproximada (3ra ley de Kepler simplificada)
            orbital_distance = (period / 365.25) ** (2/3)  # En AU
            temp = star_temp * math.sqrt(star_radius / (2 * orbital_distance))
        else:
            temp = random.uniform(200, 2000)
        
        # Parámetros de tránsito
        impact_parameter = random.uniform(0, 0.9)  # b < 1 para tránsitos observables
        
        # Duración del tránsito (fórmula aproximada)
        if star_params:
            star_radius_km = star_params.get('star_radius', 1.0) * 695700  # Radio solar en km
            planet_radius_km = radius * 6371  # Radio terrestre en km
            duration = (period * 24) / math.pi * math.sqrt((star_radius_km + planet_radius_km)**2 - (impact_parameter * star_radius_km)**2) / star_radius_km
            duration = max(duration, 0.5)  # Mínimo realista
        else:
            duration = random.uniform(1, 8)
        
        # Profundidad del tránsito (ratio de áreas)
        if star_params:
            star_radius_earth = star_params.get('star_radius', 1.0) * 109.2  # Radio solar en radios terrestres
            depth = (radius / star_radius_earth) ** 2 * 1e6  # En ppm
        else:
            depth = random.uniform(10, 5000)
        
        # Signal-to-noise ratio (realista para diferentes misiones)
        base_snr = random.uniform(7, 50)  # Threshold típico es ~7
        snr = base_snr * math.sqrt(depth / 100)  # Proporcional a sqrt(profundidad)
        
        return {
            'period': period,
            'radius': radius,
            'temp': temp,
            'impact_parameter': impact_parameter,
            'duration': duration,
            'depth': depth,
            'snr': snr,
            'planet_type': planet_type
        }
    
    def determine_disposition(self, planet_params: Dict, star_params: Dict) -> str:
        """Determina la disposición basada en parámetros físicos y criterios de detección"""
        
        # Criterios para confirmación
        snr = planet_params.get('snr', 0)
        depth = planet_params.get('depth', 0)
        period = planet_params.get('period', 0)
        duration = planet_params.get('duration', 0)
        
        # Score de calidad de detección
        quality_score = 0
        
        # SNR alto aumenta probabilidad de confirmación
        if snr > 15:
            quality_score += 3
        elif snr > 10:
            quality_score += 2
        elif snr > 7:
            quality_score += 1
        
        # Profundidad detectable
        if depth > 100:
            quality_score += 2
        elif depth > 50:
            quality_score += 1
        
        # Período en rango observable
        if 1 < period < 500:
            quality_score += 2
        elif period < 1000:
            quality_score += 1
        
        # Duración razonable
        if 1 < duration < 10:
            quality_score += 1
        
        # Determinar disposición basada en score y randomización
        rand_factor = random.random()
        
        if quality_score >= 6 and rand_factor > 0.3:
            return 'CONFIRMED'
        elif quality_score >= 4 and rand_factor > 0.5:
            return 'CANDIDATE'
        elif quality_score >= 2 and rand_factor > 0.7:
            return 'CANDIDATE'
        else:
            return 'FALSE_POSITIVE'
    
    def generate_exoplanet_data(self, count: int = 100) -> List[Dict[str, Any]]:
        """Genera un dataset completo de exoplanetas sintéticos"""
        logger.info(f"Generando {count} objetos de exoplanetas sintéticos...")
        
        exoplanets = []
        
        for i in range(count):
            # Seleccionar tipos
            stellar_type = random.choice(list(self.stellar_types.keys()))
            planet_type = random.choice(list(self.planet_types.keys()))
            
            # Generar parámetros
            star_params = self.generate_stellar_parameters(stellar_type)
            planet_params = self.generate_planetary_parameters(planet_type, star_params)
            
            # Determinar disposición
            disposition = self.determine_disposition(planet_params, star_params)
            
            # Crear registro
            exoplanet = {
                'object_id': f'SYNTHETIC-{i+1:05d}',
                'mission': random.choice(['Kepler', 'TESS', 'K2']),
                'period': planet_params['period'],
                'radius': planet_params['radius'],
                'temp': planet_params['temp'],
                'star_radius': star_params['star_radius'],
                'star_mass': star_params['star_mass'],
                'star_temp': star_params['star_temp'],
                'star_logg': star_params['star_logg'],
                'depth': planet_params['depth'],
                'duration': planet_params['duration'],
                'snr': planet_params['snr'],
                'impact_parameter': planet_params['impact_parameter'],
                'disposition': disposition,
                'planet_type': planet_params['planet_type'],
                'stellar_type': star_params['stellar_type'],
                'created_at': datetime.utcnow() - timedelta(days=random.randint(0, 365)),
                'confidence_score': random.uniform(0.1, 0.99),
                'detection_pipeline': f"Synthetic_v{random.choice(['1.0', '1.1', '2.0'])}",
                'follow_up_observations': random.randint(0, 5)
            }
            
            exoplanets.append(exoplanet)
            
            # Log progress
            if (i + 1) % 25 == 0:
                logger.info(f"Generados {i + 1}/{count} objetos...")
        
        logger.info(f"Generación completa: {count} objetos creados")
        return exoplanets


class SatelliteDataGenerator:
    """Generador de datos de satélites y misiones"""
    
    def generate_satellite_data(self) -> List[Dict[str, Any]]:
        """Genera datos de satélites/misiones espaciales"""
        
        satellites = [
            {
                'name': 'Kepler',
                'mission_type': 'Space Telescope',
                'launch_date': datetime(2009, 3, 7),
                'end_date': datetime(2017, 10, 30),
                'status': 'COMPLETED',
                'primary_mission': 'Exoplanet detection via transit photometry',
                'field_of_view': 115.6,  # degrees squared
                'targets_observed': 200000,
                'confirmed_planets': 2662,
                'planet_candidates': 4034,
                'data_quality': 0.95,
                'photometric_precision': 20,  # ppm per 6.5 hours
                'created_at': datetime.utcnow()
            },
            {
                'name': 'TESS',
                'mission_type': 'Space Telescope',
                'launch_date': datetime(2018, 4, 18),
                'end_date': None,  # Ongoing
                'status': 'ACTIVE',
                'primary_mission': 'All-sky exoplanet survey',
                'field_of_view': 2300,  # degrees squared per sector
                'targets_observed': 200000,  # Per sector
                'confirmed_planets': 350,
                'planet_candidates': 5000,
                'data_quality': 0.92,
                'photometric_precision': 60,  # ppm per hour for 10th mag star
                'created_at': datetime.utcnow()
            },
            {
                'name': 'K2',
                'mission_type': 'Space Telescope',
                'launch_date': datetime(2014, 5, 30),
                'end_date': datetime(2018, 10, 30),
                'status': 'COMPLETED',
                'primary_mission': 'Extended Kepler mission with multiple fields',
                'field_of_view': 115.6,  # degrees squared
                'targets_observed': 500000,  # Total across all campaigns
                'confirmed_planets': 500,
                'planet_candidates': 1200,
                'data_quality': 0.88,
                'photometric_precision': 80,  # ppm per 6.5 hours
                'created_at': datetime.utcnow()
            }
        ]
        
        return satellites


async def populate_mongodb():
    """Función principal para poblar MongoDB con datos de prueba"""
    
    try:
        logger.info("🚀 Iniciando población de MongoDB con datos de prueba...")
        
        # Conectar a MongoDB
        client = AsyncIOMotorClient(settings.mongodb_url)
        
        # Inicializar Beanie
        await init_beanie(
            database=client.exoplanet_db,
            document_models=[ExoplanetData, SatelliteData]
        )
        
        logger.info("✅ Conexión a MongoDB establecida")
        
        # Limpiar datos existentes (opcional)
        logger.info("🧹 Limpiando datos existentes...")
        await ExoplanetData.delete_all()
        await SatelliteData.delete_all()
        
        # Generar datos de exoplanetas
        generator = ExoplanetDataGenerator()
        exoplanet_data = generator.generate_exoplanet_data(count=500)  # 500 objetos sintéticos
        
        # Insertar datos de exoplanetas
        logger.info("📝 Insertando datos de exoplanetas...")
        exoplanet_docs = []
        for data in exoplanet_data:
            doc = ExoplanetData(**data)
            exoplanet_docs.append(doc)
        
        await ExoplanetData.insert_many(exoplanet_docs)
        
        # Generar y insertar datos de satélites
        satellite_generator = SatelliteDataGenerator()
        satellite_data = satellite_generator.generate_satellite_data()
        
        logger.info("🛰️ Insertando datos de satélites...")
        satellite_docs = []
        for data in satellite_data:
            doc = SatelliteData(**data)
            satellite_docs.append(doc)
        
        await SatelliteData.insert_many(satellite_docs)
        
        # Verificar inserción
        exoplanet_count = await ExoplanetData.count()
        satellite_count = await SatelliteData.count()
        
        logger.info(f"✅ Datos insertados exitosamente:")
        logger.info(f"   📊 Exoplanetas: {exoplanet_count}")
        logger.info(f"   🛰️ Satélites: {satellite_count}")
        
        # Estadísticas de los datos generados
        confirmed_count = await ExoplanetData.find({"disposition": "CONFIRMED"}).count()
        candidate_count = await ExoplanetData.find({"disposition": "CANDIDATE"}).count()
        false_positive_count = await ExoplanetData.find({"disposition": "FALSE_POSITIVE"}).count()
        
        logger.info(f"📈 Distribución de disposiciones:")
        logger.info(f"   ✅ Confirmados: {confirmed_count}")
        logger.info(f"   ❓ Candidatos: {candidate_count}")
        logger.info(f"   ❌ Falsos Positivos: {false_positive_count}")
        
        # Crear algunos ejemplos específicos para testing del frontend
        await create_frontend_test_examples()
        
        logger.info("🎉 Población de MongoDB completada exitosamente!")
        
        return {
            'exoplanets_inserted': exoplanet_count,
            'satellites_inserted': satellite_count,
            'confirmed_planets': confirmed_count,
            'candidates': candidate_count,
            'false_positives': false_positive_count
        }
        
    except Exception as e:
        logger.error(f"❌ Error durante la población de MongoDB: {e}")
        raise
    finally:
        # Cerrar conexión
        client.close()


async def create_frontend_test_examples():
    """Crea ejemplos específicos para testing del frontend"""
    logger.info("🧪 Creando ejemplos específicos para testing del frontend...")
    
    # Ejemplo 1: Exoplaneta confirmado tipo Earth-like
    earth_like = ExoplanetData(
        object_id="TEST-EARTH-001",
        mission="Kepler",
        period=365.25,
        radius=1.0,
        temp=288,
        star_radius=1.0,
        star_mass=1.0,
        star_temp=5778,
        star_logg=4.44,
        depth=84,  # (1 R_earth / 1 R_sun)^2 * 1e6
        duration=6.5,
        snr=25.0,
        impact_parameter=0.1,
        disposition="CONFIRMED",
        planet_type="earth_like",
        stellar_type="G_dwarf",
        confidence_score=0.95,
        detection_pipeline="Test_v1.0",
        follow_up_observations=3,
        created_at=datetime.utcnow()
    )
    
    # Ejemplo 2: Hot Jupiter candidato
    hot_jupiter = ExoplanetData(
        object_id="TEST-JUPITER-001",
        mission="TESS",
        period=3.5,
        radius=11.2,
        temp=1200,
        star_radius=1.2,
        star_mass=1.1,
        star_temp=6100,
        star_logg=4.3,
        depth=8640,  # ~(11 R_earth / 1.2 R_sun)^2 * 1e6
        duration=2.1,
        snr=45.0,
        impact_parameter=0.3,
        disposition="CANDIDATE",
        planet_type="hot_jupiter",
        stellar_type="F_dwarf",
        confidence_score=0.78,
        detection_pipeline="Test_v1.0",
        follow_up_observations=1,
        created_at=datetime.utcnow()
    )
    
    # Ejemplo 3: Falso positivo (eclipsing binary)
    false_positive = ExoplanetData(
        object_id="TEST-FALSE-001",
        mission="K2",
        period=0.8,
        radius=15.0,  # Suspiciosamente grande
        temp=2500,
        star_radius=0.8,
        star_mass=0.9,
        star_temp=4800,
        star_logg=4.6,
        depth=35000,  # Muy profundo para un planeta
        duration=0.3,  # Muy corto
        snr=8.5,
        impact_parameter=0.0,
        disposition="FALSE_POSITIVE",
        planet_type="false_positive",
        stellar_type="K_dwarf",
        confidence_score=0.15,
        detection_pipeline="Test_v1.0",
        follow_up_observations=0,
        created_at=datetime.utcnow()
    )
    
    # Insertar ejemplos
    test_examples = [earth_like, hot_jupiter, false_positive]
    await ExoplanetData.insert_many(test_examples)
    
    logger.info(f"✅ Creados {len(test_examples)} ejemplos específicos para testing")


async def verify_data():
    """Verifica que los datos se insertaron correctamente"""
    try:
        client = AsyncIOMotorClient(settings.mongodb_url)
        await init_beanie(
            database=client.exoplanet_db,
            document_models=[ExoplanetData, SatelliteData]
        )
        
        # Contar documentos
        exoplanet_count = await ExoplanetData.count()
        satellite_count = await SatelliteData.count()
        
        # Obtener algunos ejemplos
        sample_exoplanets = await ExoplanetData.find().limit(3).to_list()
        sample_satellites = await SatelliteData.find().to_list()
        
        print(f"\n📊 Verificación de datos en MongoDB:")
        print(f"   Exoplanetas: {exoplanet_count}")
        print(f"   Satélites: {satellite_count}")
        
        print(f"\n🔍 Ejemplos de exoplanetas:")
        for exo in sample_exoplanets:
            print(f"   - {exo.object_id}: {exo.disposition} ({exo.mission})")
        
        print(f"\n🛰️ Satélites disponibles:")
        for sat in sample_satellites:
            print(f"   - {sat.name}: {sat.status}")
        
        client.close()
        return True
        
    except Exception as e:
        print(f"❌ Error verificando datos: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Poblar MongoDB con datos de prueba")
    parser.add_argument("--verify", action="store_true", help="Solo verificar datos existentes")
    parser.add_argument("--count", type=int, default=500, help="Número de exoplanetas a generar")
    
    args = parser.parse_args()
    
    if args.verify:
        asyncio.run(verify_data())
    else:
        asyncio.run(populate_mongodb())