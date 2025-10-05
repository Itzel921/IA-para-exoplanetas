#!/usr/bin/env python3
"""
Script de prueba para el entrenamiento simplificado
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    print("🚀 Iniciando prueba de entrenamiento simplificado...")
    
    # Importar el simple_retrain
    from simple_retrain import SimpleTrainer
    
    print("✅ Módulo simple_retrain importado correctamente")
    
    # Crear instancia del trainer
    trainer = SimpleTrainer()
    print("✅ SimpleTrainer instanciado correctamente")
    
    # Ejecutar entrenamiento
    print("🔄 Iniciando entrenamiento...")
    trainer.train_simple_model()
    print("✅ ¡Entrenamiento completado exitosamente!")
    
except ImportError as e:
    print(f"❌ Error de importación: {e}")
except Exception as e:
    print(f"❌ Error durante el entrenamiento: {e}")
    import traceback
    traceback.print_exc()