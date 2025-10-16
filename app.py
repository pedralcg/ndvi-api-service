# gee_service/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import ee
import os
import traceback
from dotenv import load_dotenv
import tempfile 

# -------------------------------------------------------
# Configuración base
# -------------------------------------------------------
load_dotenv()

app = Flask(__name__)
# Se asegura que CORS sea amplio para /api y para /map/gee-tile (necesario para Leaflet)
CORS(app, resources={r"/api/*": {"origins": "*"}, r"/map/gee-tile/*": {"origins": "*"},})

EE_SERVICE_ACCOUNT = os.getenv("EE_SERVICE_ACCOUNT")
EE_PRIVATE_KEY_PATH = os.getenv("EE_PRIVATE_KEY_PATH", "./private-key_pedralcg.json")

# NUEVA VARIABLE DE ENTORNO PARA DESPLIEGUE EN RENDER
EE_PRIVATE_KEY_CONTENT = os.getenv("EE_PRIVATE_KEY_CONTENT")


# -------------------------------------------------------
# Inicialización de Earth Engine (Lógica HÍBRIDA)
# -------------------------------------------------------
# Declaración de la variable a nivel de módulo
TEMP_KEY_FILE = None 

try:
    if EE_SERVICE_ACCOUNT:
        key_path = None
        
        # PRIORIDAD 1: PRODUCCIÓN (RENDER) - Usar el contenido de la clave
        if EE_PRIVATE_KEY_CONTENT:
            # Crea un archivo temporal para que ee.ServiceAccountCredentials lo pueda leer
            with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.json') as tmp_file:
                tmp_file.write(EE_PRIVATE_KEY_CONTENT)
                key_path = tmp_file.name
                # Asignación simple, ya que está definida globalmente
                TEMP_KEY_FILE = key_path 
            
            print("⚙️ Usando contenido de clave (Render). Archivo temporal creado.")
        
        # PRIORIDAD 2: LOCAL - Usar la ruta del archivo local
        elif os.path.exists(EE_PRIVATE_KEY_PATH):
            key_path = EE_PRIVATE_KEY_PATH
            print("⚙️ Usando ruta de clave local (Desarrollo).")

        
        # INICIALIZACIÓN FINAL
        if key_path:
            credentials = ee.ServiceAccountCredentials(EE_SERVICE_ACCOUNT, key_path)
            ee.Initialize(credentials)
            print("✅ GEE inicializado con ServiceAccountCredentials.")
        else:
            # Si no hay clave, pero hay cuenta, intentamos inicialización por defecto
            ee.Initialize() 
            print("✅ GEE inicializado con credenciales por defecto (Puede fallar en Render).")

    else:
        # Si no hay service account, intentamos inicialización por defecto
        ee.Initialize()
        print("✅ GEE inicializado con credenciales por defecto.")
        
except Exception as e:
    print("⚠️ Falló la inicialización de GEE:", e)
    traceback.print_exc()

# Limpieza del archivo temporal si fue creado
finally:
    # Ahora TEMP_KEY_FILE está disponible globalmente
    if TEMP_KEY_FILE and os.path.exists(TEMP_KEY_FILE):
        try:
            os.remove(TEMP_KEY_FILE)
            print("🗑️ Archivo temporal de clave eliminado.")
        except Exception as e:
            print(f"⚠️ No se pudo eliminar el archivo temporal {TEMP_KEY_FILE}: {e}")

# -------------------------------------------------------
# Funciones GEE Helpers
# -------------------------------------------------------

def add_ndvi(image):
    # NDVI = (Banda 8 - Banda 4) / (Banda 8 + Banda 4)
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("ndvi")
    return image.addBands(ndvi)

def mask_s2_clouds(image):
    scl = image.select("SCL")
    cloud_shadow_mask = scl.neq(3)
    cloud_mask = scl.neq(8).And(scl.neq(9)).And(scl.neq(10))
    mask = cloud_shadow_mask.And(cloud_mask)
    return image.updateMask(mask).copyProperties(image, ["system:time_start"])

# -------------------------------------------------------
# Helper CRÍTICO: construir geometría y aplicar buffer extra de 1km
# -------------------------------------------------------
def build_buffered_aoi(geojson_feature):
    """
    Crea una ee.Geometry aplicando un buffer adicional de 1000m (1km) 
    a todas las AOI. Si es un círculo, suma 1000m a su radio original.
    """
    
    # Asumimos que recibimos el GeoJSON Feature completo del frontend
    geometry_data = geojson_feature.get("geometry")
    properties = geojson_feature.get("properties", {})
    geom_type = geometry_data.get("type", "").lower()

    try:
        # --- Caso 1: Círculo de Leaflet (Point con radio en properties) ---
        if geom_type == "point" and "radius" in properties:
            radius_m = properties["radius"]
            coords = geometry_data["coordinates"]
            point_ee = ee.Geometry.Point(coords)
            
            # CRÍTICO: Buffer = Radio original + 1000m extra
            final_buffer = radius_m + 1000
            geom_aoi = point_ee.buffer(final_buffer)
            print(f"⚪ Círculo detectado: Buffer = {radius_m}m original + 1000m extra = {final_buffer}m")
        
        # --- Caso 2: Punto simple (Marker), Polígono, Rectángulo, etc. ---
        else:
            # Crear la geometría original
            geom_original = ee.Geometry(geometry_data)
            
            # CRÍTICO: Aplicar buffer de 1000m a la geometría original
            geom_aoi = geom_original.buffer(1000)
            print(f"🟦 {geom_type.capitalize()} detectado: Buffer de 1000m aplicado.")

    except Exception as e:
        print(f"⚠️ Error al construir geometría y buffer: {e}. Usando geometría sin buffer de 1km.")
        geom_aoi = ee.Geometry(geometry_data)

    return geom_aoi


# -------------------------------------------------------
# Endpoint API para calcular NDVI
# -------------------------------------------------------
@app.route("/api/ndvi", methods=["POST"])
def calculate_ndvi():
    try:
        data = request.json
        date_str = data.get("date")
        # El frontend ahora envía el GeoJSON Feature completo
        geojson_feature = data.get("geometry") 

        if not date_str or not geojson_feature:
            return jsonify({"status": "error", "message": "Faltan parámetros: 'date' o 'geometry'."}), 400

        # --- 1. Determinar la Geometría de Earth Engine (con buffer de 1km extra) ---
        geometry_ee_clip_reduce = build_buffered_aoi(geojson_feature)
        
        # Calcular el área en km2 (del AOI final, con buffer)
        area_km2 = geometry_ee_clip_reduce.area().divide(1e6).getInfo()

        # --- 2. Colección de Imágenes y Filtros ---
        date_ee = ee.Date(date_str)
        
        col = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")\
            .filterBounds(geometry_ee_clip_reduce)\
            .filterDate(date_ee.advance(-1, 'month'), date_ee.advance(1, 'month'))\
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))\
            .map(mask_s2_clouds)\
            .map(add_ndvi)
        
        imgs_found = col.size().getInfo()
        
        if imgs_found == 0:
            return jsonify({
                "status": "warning",
                "message": "⚠️ No se encontraron imágenes Sentinel-2 disponibles cerca de esa fecha y área.",
                "images_found": 0,
            }), 200

        # Obtener la imagen más cercana
        def add_diff(img):
            diff = ee.Number(img.date().difference(date_ee, "day")).abs()
            return img.set("diff", diff)

        nearest = ee.Image(col.map(add_diff).sort("diff").first())
        
        # --- 3. Obtener Valor de NDVI y URL de Mosaico ---
        
        mean_val = nearest.reduceRegion(
            reducer=ee.Reducer.mean(),
            # Usar el AOI con buffer para el cálculo
            geometry=geometry_ee_clip_reduce, 
            scale=10, 
            maxPixels=1e13,
            bestEffort=True 
        )
        mean_val_numeric = mean_val.get("ndvi").getInfo()
        
        vis_params = {
            "min": 0.0,
            "max": 1.0,
            "palette": [
                "FFFFFF", "CE7E45", "DF923D", "F1B555", "FCD163", 
                "99B718", "74A901", "66A000", "529400", "3E8601", 
                "207401", "056201", "004C00", "002C00", "001500",
            ],
        }

        # Aplicar clip a la imagen NDVI usando el AOI con buffer para visualización
        clipped_ndvi = nearest.select("ndvi").clip(geometry_ee_clip_reduce)

        # Obtener la URL del mosaico recortado
        tile_url = clipped_ndvi.getMapId(vis_params)["tile_fetcher"].url_format
        
        # --- 4. Obtener Información Adicional (Bounds y Fecha) ---
        try:
            bounds_obj = geometry_ee_clip_reduce.bounds().getInfo()
            coords = bounds_obj["coordinates"][0]
            lats = [c[1] for c in coords]
            lons = [c[0] for c in coords]
            bounds = [[min(lats), min(lons)], [max(lats), max(lons)]]
        except Exception:
            bounds = None

        image_date = nearest.date().format("YYYY-MM-dd").getInfo()

        response = {
            "status": "success",
            "mean_ndvi": round(mean_val_numeric, 4) if mean_val_numeric is not None else None,
            "tile_url": tile_url,
            "bounds": bounds,
            "images_found": imgs_found,
            "image_date": image_date,
            "area_km2": area_km2,
        }

        print("✅ NDVI generado correctamente (CLIP aplicado):", image_date)
        return jsonify(response)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"Error interno del servidor GEE: {e}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
