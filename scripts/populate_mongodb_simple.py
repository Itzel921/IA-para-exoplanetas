#!/usr/bin/env python3
"""
Script simple para poblar MongoDB con datos de prueba
NASA Space Apps Challenge 2025
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any
import random
import math

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Datos de prueba sintéticos simples
SYNTHETIC_EXOPLANETS = [
    {
        "object_id": "TEST-EARTH-001",
        "mission": "Kepler",
        "period": 365.25,
        "radius": 1.0,
        "temp": 288,
        "star_radius": 1.0,
        "star_mass": 1.0,
        "star_temp": 5778,
        "star_logg": 4.44,
        "depth": 84,
        "duration": 6.5,
        "snr": 25.0,
        "impact_parameter": 0.1,
        "disposition": "CONFIRMED",
        "planet_type": "earth_like",
        "stellar_type": "G_dwarf",
        "confidence_score": 0.95,
        "detection_pipeline": "Test_v1.0",
        "follow_up_observations": 3,
        "created_at": datetime.utcnow().isoformat()
    },
    {
        "object_id": "TEST-JUPITER-001",
        "mission": "TESS",
        "period": 3.5,
        "radius": 11.2,
        "temp": 1200,
        "star_radius": 1.2,
        "star_mass": 1.1,
        "star_temp": 6100,
        "star_logg": 4.3,
        "depth": 8640,
        "duration": 2.1,
        "snr": 45.0,
        "impact_parameter": 0.3,
        "disposition": "CANDIDATE",
        "planet_type": "hot_jupiter",
        "stellar_type": "F_dwarf",
        "confidence_score": 0.78,
        "detection_pipeline": "Test_v1.0",
        "follow_up_observations": 1,
        "created_at": datetime.utcnow().isoformat()
    },
    {
        "object_id": "TEST-FALSE-001",
        "mission": "K2",
        "period": 0.8,
        "radius": 15.0,
        "temp": 2500,
        "star_radius": 0.8,
        "star_mass": 0.9,
        "star_temp": 4800,
        "star_logg": 4.6,
        "depth": 35000,
        "duration": 0.3,
        "snr": 8.5,
        "impact_parameter": 0.0,
        "disposition": "FALSE_POSITIVE",
        "planet_type": "false_positive",
        "stellar_type": "K_dwarf",
        "confidence_score": 0.15,
        "detection_pipeline": "Test_v1.0",
        "follow_up_observations": 0,
        "created_at": datetime.utcnow().isoformat()
    }
]

def generate_synthetic_exoplanet(index: int) -> Dict[str, Any]:
    """Genera un exoplaneta sintético realista"""
    
    # Tipos de planetas y estrellas
    planet_types = ["earth_like", "super_earth", "mini_neptune", "hot_jupiter", "cold_jupiter"]
    stellar_types = ["M_dwarf", "K_dwarf", "G_dwarf", "F_dwarf", "A_dwarf"]
    missions = ["Kepler", "TESS", "K2"]
    
    planet_type = random.choice(planet_types)
    stellar_type = random.choice(stellar_types)
    mission = random.choice(missions)
    
    # Generar parámetros realistas
    if planet_type == "earth_like":
        radius = random.uniform(0.8, 1.2)
        period = random.uniform(200, 500)
        temp = random.uniform(250, 350)
    elif planet_type == "super_earth":
        radius = random.uniform(1.2, 2.0)
        period = random.uniform(10, 200)
        temp = random.uniform(300, 800)
    elif planet_type == "mini_neptune":
        radius = random.uniform(2.0, 4.0)
        period = random.uniform(5, 100)
        temp = random.uniform(400, 1000)
    elif planet_type == "hot_jupiter":
        radius = random.uniform(8.0, 15.0)
        period = random.uniform(1, 10)
        temp = random.uniform(1000, 2500)
    else:  # cold_jupiter
        radius = random.uniform(10.0, 20.0)
        period = random.uniform(100, 2000)
        temp = random.uniform(100, 300)
    
    # Parámetros estelares
    if stellar_type == "M_dwarf":
        star_temp = random.uniform(2300, 3800)
        star_radius = random.uniform(0.1, 0.6)
        star_mass = random.uniform(0.08, 0.6)
    elif stellar_type == "K_dwarf":
        star_temp = random.uniform(3700, 5200)
        star_radius = random.uniform(0.6, 0.9)
        star_mass = random.uniform(0.6, 0.9)
    elif stellar_type == "G_dwarf":
        star_temp = random.uniform(5200, 6000)
        star_radius = random.uniform(0.8, 1.2)
        star_mass = random.uniform(0.8, 1.2)
    elif stellar_type == "F_dwarf":
        star_temp = random.uniform(6000, 7500)
        star_radius = random.uniform(1.0, 1.5)
        star_mass = random.uniform(1.0, 1.5)
    else:  # A_dwarf
        star_temp = random.uniform(7500, 10000)
        star_radius = random.uniform(1.4, 2.5)
        star_mass = random.uniform(1.4, 2.5)
    
    star_logg = math.log10(star_mass / (star_radius ** 2)) + 4.44
    
    # Parámetros de tránsito
    depth = (radius / (star_radius * 109.2)) ** 2 * 1e6  # ppm
    duration = random.uniform(1, 8)
    snr = random.uniform(7, 50)
    impact_parameter = random.uniform(0, 0.9)
    
    # Determinar disposición
    quality_score = 0
    if snr > 15: quality_score += 3
    elif snr > 10: quality_score += 2
    elif snr > 7: quality_score += 1
    
    if depth > 100: quality_score += 2
    elif depth > 50: quality_score += 1
    
    if 1 < period < 500: quality_score += 2
    elif period < 1000: quality_score += 1
    
    rand_factor = random.random()
    if quality_score >= 6 and rand_factor > 0.3:
        disposition = 'CONFIRMED'
    elif quality_score >= 4 and rand_factor > 0.5:
        disposition = 'CANDIDATE'
    elif quality_score >= 2 and rand_factor > 0.7:
        disposition = 'CANDIDATE'
    else:
        disposition = 'FALSE_POSITIVE'
    
    return {
        "object_id": f"SYNTHETIC-{index+1:05d}",
        "mission": mission,
        "period": round(period, 3),
        "radius": round(radius, 3),
        "temp": round(temp, 1),
        "star_radius": round(star_radius, 3),
        "star_mass": round(star_mass, 3),
        "star_temp": round(star_temp, 1),
        "star_logg": round(star_logg, 3),
        "depth": round(depth, 1),
        "duration": round(duration, 2),
        "snr": round(snr, 1),
        "impact_parameter": round(impact_parameter, 2),
        "disposition": disposition,
        "planet_type": planet_type,
        "stellar_type": stellar_type,
        "confidence_score": round(random.uniform(0.1, 0.99), 3),
        "detection_pipeline": f"Synthetic_v{random.choice(['1.0', '1.1', '2.0'])}",
        "follow_up_observations": random.randint(0, 5),
        "created_at": (datetime.utcnow() - timedelta(days=random.randint(0, 365))).isoformat()
    }

def generate_satellite_data() -> List[Dict[str, Any]]:
    """Genera datos de satélites/misiones"""
    return [
        {
            "name": "Kepler",
            "mission_type": "Space Telescope",
            "launch_date": "2009-03-07",
            "end_date": "2017-10-30",
            "status": "COMPLETED",
            "primary_mission": "Exoplanet detection via transit photometry",
            "field_of_view": 115.6,
            "targets_observed": 200000,
            "confirmed_planets": 2662,
            "planet_candidates": 4034,
            "data_quality": 0.95,
            "photometric_precision": 20,
            "created_at": datetime.utcnow().isoformat()
        },
        {
            "name": "TESS",
            "mission_type": "Space Telescope",
            "launch_date": "2018-04-18",
            "end_date": None,
            "status": "ACTIVE",
            "primary_mission": "All-sky exoplanet survey",
            "field_of_view": 2300,
            "targets_observed": 200000,
            "confirmed_planets": 350,
            "planet_candidates": 5000,
            "data_quality": 0.92,
            "photometric_precision": 60,
            "created_at": datetime.utcnow().isoformat()
        },
        {
            "name": "K2",
            "mission_type": "Space Telescope",
            "launch_date": "2014-05-30",
            "end_date": "2018-10-30",
            "status": "COMPLETED",
            "primary_mission": "Extended Kepler mission with multiple fields",
            "field_of_view": 115.6,
            "targets_observed": 500000,
            "confirmed_planets": 500,
            "planet_candidates": 1200,
            "data_quality": 0.88,
            "photometric_precision": 80,
            "created_at": datetime.utcnow().isoformat()
        }
    ]

async def populate_mongodb_simple():
    """Pobla MongoDB usando PyMongo directamente"""
    try:
        import pymongo
        from pymongo import MongoClient
        
        logger.info("🚀 Iniciando población simple de MongoDB...")
        
        # Conectar a MongoDB
        client = MongoClient("mongodb://localhost:27017")
        db = client.exoplanet_db
        
        # Limpiar colecciones existentes
        db.exoplanet_data.drop()
        db.satellite_data.drop()
        
        logger.info("🧹 Colecciones limpiadas")
        
        # Insertar datos de exoplanetas (algunos predefinidos + sintéticos)
        all_exoplanets = SYNTHETIC_EXOPLANETS.copy()
        
        # Generar más datos sintéticos
        for i in range(100):  # 100 objetos adicionales
            synthetic = generate_synthetic_exoplanet(i + 3)  # +3 por los predefinidos
            all_exoplanets.append(synthetic)
        
        # Insertar exoplanetas
        result_exo = db.exoplanet_data.insert_many(all_exoplanets)
        logger.info(f"📊 Insertados {len(result_exo.inserted_ids)} exoplanetas")
        
        # Insertar datos de satélites
        satellite_data = generate_satellite_data()
        result_sat = db.satellite_data.insert_many(satellite_data)
        logger.info(f"🛰️ Insertados {len(result_sat.inserted_ids)} satélites")
        
        # Verificar inserción
        exo_count = db.exoplanet_data.count_documents({})
        sat_count = db.satellite_data.count_documents({})
        
        # Estadísticas
        confirmed = db.exoplanet_data.count_documents({"disposition": "CONFIRMED"})
        candidates = db.exoplanet_data.count_documents({"disposition": "CANDIDATE"})
        false_positives = db.exoplanet_data.count_documents({"disposition": "FALSE_POSITIVE"})
        
        logger.info(f"✅ Datos insertados exitosamente:")
        logger.info(f"   📊 Total exoplanetas: {exo_count}")
        logger.info(f"   🛰️ Total satélites: {sat_count}")
        logger.info(f"   ✅ Confirmados: {confirmed}")
        logger.info(f"   ❓ Candidatos: {candidates}")
        logger.info(f"   ❌ Falsos Positivos: {false_positives}")
        
        # Mostrar algunos ejemplos
        logger.info("🔍 Ejemplos insertados:")
        examples = db.exoplanet_data.find().limit(3)
        for i, exo in enumerate(examples, 1):
            logger.info(f"   {i}. {exo['object_id']}: {exo['disposition']} ({exo['mission']})")
        
        client.close()
        logger.info("🎉 Población completada exitosamente!")
        
        return {
            'exoplanets_inserted': exo_count,
            'satellites_inserted': sat_count,
            'confirmed_planets': confirmed,
            'candidates': candidates,
            'false_positives': false_positives
        }
        
    except Exception as e:
        logger.error(f"❌ Error durante la población: {e}")
        raise

async def verify_data_simple():
    """Verifica los datos en MongoDB"""
    try:
        import pymongo
        from pymongo import MongoClient
        
        client = MongoClient("mongodb://localhost:27017")
        db = client.exoplanet_db
        
        exo_count = db.exoplanet_data.count_documents({})
        sat_count = db.satellite_data.count_documents({})
        
        print(f"\n📊 Verificación de datos en MongoDB:")
        print(f"   Exoplanetas: {exo_count}")
        print(f"   Satélites: {sat_count}")
        
        if exo_count > 0:
            print(f"\n🔍 Ejemplos de exoplanetas:")
            examples = db.exoplanet_data.find().limit(5)
            for exo in examples:
                print(f"   - {exo['object_id']}: {exo['disposition']} ({exo['mission']})")
        
        if sat_count > 0:
            print(f"\n🛰️ Satélites disponibles:")
            satellites = db.satellite_data.find()
            for sat in satellites:
                print(f"   - {sat['name']}: {sat['status']}")
        
        client.close()
        return True
        
    except Exception as e:
        print(f"❌ Error verificando datos: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Poblar MongoDB con datos de prueba (versión simple)")
    parser.add_argument("--verify", action="store_true", help="Solo verificar datos existentes")
    
    args = parser.parse_args()
    
    if args.verify:
        asyncio.run(verify_data_simple())
    else:
        asyncio.run(populate_mongodb_simple())