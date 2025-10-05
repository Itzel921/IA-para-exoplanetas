import pandas as pd
from pathlib import Path

# Función especializada para leer archivos CSV de NASA
def load_nasa_csv(file_path, max_attempts=5):
    """
    Carga archivos CSV de NASA que pueden tener metadatos o formato inconsistente
    """
    print(f"🔄 Intentando cargar: {Path(file_path).name}")
    
    # Método 1: Saltar líneas de comentarios (común en archivos NASA)
    for skip_rows in range(max_attempts):
        try:
            df = pd.read_csv(file_path, 
                           skiprows=skip_rows,
                           on_bad_lines='skip',
                           engine='python')
            
            # Verificar que tenemos datos válidos
            if len(df) > 0 and len(df.columns) > 5:
                print(f"✅ Éxito saltando {skip_rows} filas iniciales")
                print(f"📊 Dimensiones: {df.shape}")
                return df
                
        except Exception as e:
            print(f"❌ Intento {skip_rows + 1} falló: {str(e)[:50]}...")
            continue
    
    # Método 2: Detectar automáticamente el delimitador
    try:
        import csv
        with open(file_path, 'r', encoding='utf-8') as file:
            sample = file.read(1024)
            dialect = csv.Sniffer().sniff(sample)
            
        df = pd.read_csv(file_path, 
                        delimiter=dialect.delimiter,
                        on_bad_lines='skip',
                        engine='python')
        print(f"✅ Éxito con delimitador automático: '{dialect.delimiter}'")
        return df
        
    except Exception as e:
        print(f"❌ Detección automática falló: {e}")
    
    # Método 3: Fuerza bruta - diferentes encodings
    encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path,
                           encoding=encoding,
                           on_bad_lines='skip',
                           engine='python')
            print(f"✅ Éxito con encoding: {encoding}")
            return df
        except Exception:
            continue
    
    print("🚨 Todos los métodos fallaron")
    return None

# Usar la función
current_dir = Path(__file__).parent
data_path = current_dir.parent / "data" / "datasets" / "cumulative_2025.10.04_11.46.06.csv"

df = load_nasa_csv(data_path)

if df is not None:
    print(f"\n📋 Información del dataset:")
    print(f"   Filas: {df.shape[0]:,}")
    print(f"   Columnas: {df.shape[1]}")
    print(f"\n🔤 Primeras 5 columnas:")
    for i, col in enumerate(df.columns[:5]):
        print(f"   {i+1}. {col}")
    
    print(f"\n📊 Muestra de datos:")
    print(df.head(3))
else:
    print("No se pudo cargar el archivo.")