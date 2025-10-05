"""
Servicio de manejo de archivos
Gestiona subida, procesamiento y descarga de archivos
"""

import os
import shutil
import aiofiles
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd
import logging
from fastapi import UploadFile

logger = logging.getLogger(__name__)

class FileService:
    """
    Servicio para manejo de archivos de la aplicación
    """
    
    def __init__(self, upload_path: Path, results_path: Path):
        self.upload_path = upload_path
        self.results_path = results_path
        
        # Crear directorios si no existen
        self.upload_path.mkdir(parents=True, exist_ok=True)
        self.results_path.mkdir(parents=True, exist_ok=True)
        
        # Configurar límites
        self.max_file_size = 100 * 1024 * 1024  # 100MB
        self.allowed_extensions = {'.csv', '.txt'}
        
        logger.info(f"📁 FileService inicializado:")
        logger.info(f"   • Upload path: {self.upload_path}")
        logger.info(f"   • Results path: {self.results_path}")
    
    async def save_upload_file(self, upload_file: UploadFile) -> Path:
        """
        Guarda un archivo subido temporalmente
        """
        try:
            # Validar archivo
            await self._validate_upload_file(upload_file)
            
            # Generar nombre único
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_filename = self._sanitize_filename(upload_file.filename)
            temp_filename = f"{timestamp}_{safe_filename}"
            temp_path = self.upload_path / temp_filename
            
            logger.info(f"💾 Guardando archivo: {temp_filename}")
            
            # Guardar archivo
            async with aiofiles.open(temp_path, 'wb') as f:
                content = await upload_file.read()
                await f.write(content)
            
            # Validar contenido CSV
            await self._validate_csv_content(temp_path)
            
            logger.info(f"✅ Archivo guardado: {temp_path.name} ({len(content)} bytes)")
            return temp_path
            
        except Exception as e:
            logger.error(f"❌ Error guardando archivo: {e}")
            raise
    
    async def _validate_upload_file(self, upload_file: UploadFile):
        """Validar archivo subido"""
        
        # Verificar extensión
        file_ext = Path(upload_file.filename).suffix.lower()
        if file_ext not in self.allowed_extensions:
            raise ValueError(f"Extensión no permitida: {file_ext}. Permitidas: {self.allowed_extensions}")
        
        # Verificar tipo de contenido
        if upload_file.content_type not in ['text/csv', 'application/csv', 'text/plain']:
            logger.warning(f"Content-type inesperado: {upload_file.content_type}")
        
        # Verificar que no esté vacío
        if upload_file.size == 0:
            raise ValueError("El archivo está vacío")
        
        # Verificar tamaño
        if upload_file.size > self.max_file_size:
            raise ValueError(f"Archivo demasiado grande: {upload_file.size} bytes (máximo: {self.max_file_size})")
    
    async def _validate_csv_content(self, file_path: Path):
        """Validar contenido del CSV"""
        try:
            # Intentar leer el CSV
            df = pd.read_csv(file_path, nrows=5)  # Solo las primeras 5 filas para validar
            
            if len(df) == 0:
                raise ValueError("El archivo CSV está vacío")
            
            if len(df.columns) == 0:
                raise ValueError("El archivo CSV no tiene columnas")
            
            logger.info(f"📊 CSV válido: {len(df.columns)} columnas detectadas")
            
        except pd.errors.EmptyDataError:
            raise ValueError("El archivo CSV está vacío o malformado")
        except pd.errors.ParserError as e:
            raise ValueError(f"Error parseando CSV: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error validando CSV: {str(e)}")
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitizar nombre de archivo"""
        # Remover caracteres problemáticos
        import re
        
        # Mantener solo caracteres alfanuméricos, puntos, guiones y guiones bajos
        safe_name = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
        
        # Limitar longitud
        if len(safe_name) > 100:
            name_part = Path(safe_name).stem[:90]
            ext_part = Path(safe_name).suffix
            safe_name = f"{name_part}{ext_part}"
        
        return safe_name
    
    async def cleanup_temp_file(self, file_path: Path):
        """Limpiar archivo temporal"""
        try:
            if file_path.exists():
                file_path.unlink()
                logger.info(f"🗑️ Archivo temporal eliminado: {file_path.name}")
        except Exception as e:
            logger.warning(f"⚠️ Error eliminando archivo temporal: {e}")
    
    async def list_result_files(self) -> List[Dict[str, Any]]:
        """Listar archivos de resultados disponibles"""
        try:
            result_files = []
            
            # Buscar archivos CSV de resultados
            csv_files = list(self.results_path.glob("*.csv"))
            json_files = list(self.results_path.glob("*.json"))
            
            all_files = csv_files + json_files
            
            for file_path in all_files:
                try:
                    stat = file_path.stat()
                    
                    file_info = {
                        "filename": file_path.name,
                        "size": stat.st_size,
                        "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "type": "predictions" if "prediction" in file_path.name else "summary",
                        "extension": file_path.suffix
                    }
                    
                    result_files.append(file_info)
                    
                except Exception as e:
                    logger.warning(f"Error obteniendo info de {file_path.name}: {e}")
            
            # Ordenar por fecha de modificación (más reciente primero)
            result_files.sort(key=lambda x: x["modified"], reverse=True)
            
            logger.info(f"📋 Encontrados {len(result_files)} archivos de resultados")
            return result_files
            
        except Exception as e:
            logger.error(f"❌ Error listando archivos de resultados: {e}")
            return []
    
    async def get_file_info(self, filename: str) -> Dict[str, Any]:
        """Obtener información detallada de un archivo"""
        try:
            file_path = self.results_path / filename
            
            if not file_path.exists():
                raise FileNotFoundError(f"Archivo no encontrado: {filename}")
            
            stat = file_path.stat()
            
            info = {
                "filename": filename,
                "size": stat.st_size,
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "extension": file_path.suffix,
                "path": str(file_path)
            }
            
            # Si es CSV, obtener información adicional
            if file_path.suffix.lower() == '.csv':
                try:
                    df = pd.read_csv(file_path, nrows=1)
                    info["columns"] = len(df.columns)
                    info["sample_columns"] = df.columns.tolist()[:10]  # Primeras 10 columnas
                    
                    # Contar filas totales de forma eficiente
                    with open(file_path, 'r') as f:
                        row_count = sum(1 for line in f) - 1  # -1 por el header
                    info["rows"] = row_count
                    
                except Exception:
                    pass
            
            return info
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo info del archivo {filename}: {e}")
            raise
    
    async def cleanup_old_files(self, days_old: int = 7):
        """Limpiar archivos antiguos"""
        try:
            cutoff_time = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
            cleaned_count = 0
            
            # Limpiar archivos temporales
            for file_path in self.upload_path.glob("*"):
                if file_path.stat().st_ctime < cutoff_time:
                    file_path.unlink()
                    cleaned_count += 1
            
            if cleaned_count > 0:
                logger.info(f"🧹 Limpiados {cleaned_count} archivos antiguos")
            
        except Exception as e:
            logger.error(f"❌ Error limpiando archivos antiguos: {e}")
    
    def get_upload_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de uploads"""
        try:
            upload_files = list(self.upload_path.glob("*"))
            result_files = list(self.results_path.glob("*"))
            
            upload_size = sum(f.stat().st_size for f in upload_files if f.is_file())
            result_size = sum(f.stat().st_size for f in result_files if f.is_file())
            
            stats = {
                "upload_files_count": len(upload_files),
                "result_files_count": len(result_files),
                "upload_size_mb": round(upload_size / (1024 * 1024), 2),
                "result_size_mb": round(result_size / (1024 * 1024), 2),
                "total_size_mb": round((upload_size + result_size) / (1024 * 1024), 2)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas: {e}")
            return {}