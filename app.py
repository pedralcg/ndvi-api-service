# gee_service/app.py - Backend actualizado con buffers modificados y datos mejorados
from flask import Flask, request, jsonify
from flask_cors import CORS
import ee
import os
import traceback
from dotenv import load_dotenv
import tempfile
import re


# -------------------------------------------------------
# Configuración base
# -------------------------------------------------------
load_dotenv()

app = Flask(__name__)

# CORS MEJORADO - Soporta OPTIONS preflight
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Añadir manejo explícito de OPTIONS para todos los endpoints
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

# -------------------------------------------------------
# Variables de entorno
# -------------------------------------------------------
EE_SERVICE_ACCOUNT = os.getenv("EE_SERVICE_ACCOUNT")
EE_PRIVATE_KEY_PATH = os.getenv("EE_PRIVATE_KEY_PATH", "./private-key_pedralcg.json")
EE_PRIVATE_KEY_CONTENT = os.getenv("EE_PRIVATE_KEY_CONTENT")

# -------------------------------------------------------
# Inicialización de Earth Engine
# -------------------------------------------------------
TEMP_KEY_FILE = None

try:
    if EE_SERVICE_ACCOUNT:
        key_path = None
        
        if EE_PRIVATE_KEY_CONTENT:
            with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.json') as tmp_file:
                tmp_file.write(EE_PRIVATE_KEY_CONTENT)
                key_path = tmp_file.name
                TEMP_KEY_FILE = key_path
            print("⚙️ Usando contenido de clave (Render).")
        elif os.path.exists(EE_PRIVATE_KEY_PATH):
            key_path = EE_PRIVATE_KEY_PATH
            print("⚙️ Usando ruta de clave local.")
        
        if key_path:
            credentials = ee.ServiceAccountCredentials(EE_SERVICE_ACCOUNT, key_path)
            ee.Initialize(credentials)
            print("✅ GEE inicializado con ServiceAccountCredentials.")
        else:
            ee.Initialize()
            print("✅ GEE inicializado por defecto.")
    else:
        ee.Initialize()
        print("✅ GEE inicializado por defecto.")
        
except Exception as e:
    print("⚠️ Falló la inicialización de GEE:", e)
    traceback.print_exc()

finally:
    if TEMP_KEY_FILE and os.path.exists(TEMP_KEY_FILE):
        try:
            os.remove(TEMP_KEY_FILE)
            print("🗑️ Archivo temporal eliminado.")
        except Exception as e:
            print(f"⚠️ No se pudo eliminar: {e}")

# -------------------------------------------------------
# Funciones auxiliares comunes
# -------------------------------------------------------

def mask_s2_clouds(image):
    """Máscara de nubes y sombras para Sentinel-2"""
    scl = image.select("SCL")
    cloud_shadow_mask = scl.neq(3)
    cloud_mask = scl.neq(8).And(scl.neq(9)).And(scl.neq(10))
    mask = cloud_shadow_mask.And(cloud_mask)
    return image.updateMask(mask).copyProperties(image, ["system:time_start"])


def build_geometry_aoi(geojson_feature):
    """
    ACTUALIZADO: Nueva lógica de buffers
    - Punto: buffer de 5 metros ÚNICAMENTE
    - Polígonos, rectángulos, círculos: SIN buffer
    """
    geometry_data = geojson_feature.get("geometry")
    properties = geojson_feature.get("properties", {})
    geom_type = geometry_data.get("type", "").lower()

    try:
        if geom_type == "point":
            coords = geometry_data["coordinates"]
            point_ee = ee.Geometry.Point(coords)
            
            # Si tiene radius (círculo), usar ese radio SIN buffer adicional
            if "radius" in properties:
                radius_m = properties["radius"]
                geom_aoi = point_ee.buffer(radius_m)
                print(f"⚪ Círculo: Radio = {radius_m}m (sin buffer adicional)")
            else:
                # Punto simple: buffer de 5m
                geom_aoi = point_ee.buffer(5)
                print(f"📍 Punto: Buffer = 5m")
        else:
            # Polígonos, rectángulos, LineString: SIN buffer
            geom_aoi = ee.Geometry(geometry_data)
            print(f"🟦 {geom_type.capitalize()}: Sin buffer")
    except Exception as e:
        print(f"⚠️ Error al construir geometría: {e}")
        traceback.print_exc()
        geom_aoi = ee.Geometry(geometry_data)

    return geom_aoi


def add_spectral_index(image, index_name):
    """Añade un índice espectral específico a la imagen"""
    if index_name == 'NDVI':
        return image.addBands(
            image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        )
    elif index_name == 'NBR':
        return image.addBands(
            image.normalizedDifference(['B8', 'B12']).rename('NBR')
        )
    elif index_name == 'CIre':
        return image.addBands(
            image.expression('(nir/re1)-1', {
                'nir': image.select('B8'),
                're1': image.select('B5')
            }).rename('CIre')
        )
    elif index_name == 'MSI':
        return image.addBands(
            image.expression('swir/nir', {
                'swir': image.select('B11'),
                'nir': image.select('B8')
            }).rename('MSI')
        )
    return image


def add_all_indices(image):
    """Añade todos los índices espectrales a la imagen"""
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    nbr = image.normalizedDifference(['B8', 'B12']).rename('NBR')
    cire = image.expression('(nir/re1)-1', {
        'nir': image.select('B8'),
        're1': image.select('B5')
    }).rename('CIre')
    msi = image.expression('swir/nir', {
        'swir': image.select('B11'),
        'nir': image.select('B8')
    }).rename('MSI')
    return image.addBands([ndvi, nbr, cire, msi])


def get_image_dates_info(image_collection):
    """
    NUEVO: Extrae información detallada de las imágenes en la colección
    """
    def get_date_info(img):
        return ee.Feature(None, {
            'date': ee.Date(img.get('system:time_start')).format('YYYY-MM-dd'),
            'cloud_percentage': img.get('CLOUDY_PIXEL_PERCENTAGE'),
            'tile': img.get('MGRS_TILE')
        })
    
    features = image_collection.map(get_date_info).getInfo()['features']
    return [
        {
            'date': f['properties']['date'],
            'cloud_percentage': round(f['properties']['cloud_percentage'], 2) if f['properties']['cloud_percentage'] else None,
            'tile': f['properties']['tile']
        }
        for f in features
    ]


def get_visualization_params(index_name):
    """Retorna parámetros de visualización para cada índice"""
    params = {
        'NDVI': {
            'min': -0.2,
            'max': 1.0,
            'palette': [
                "FFFFFF", "CE7E45", "DF923D", "F1B555", "FCD163",
                "99B718", "74A901", "66A000", "529400", "3E8601",
                "207401", "056201", "004C00", "002C00", "001500"
            ]
        },
        'NBR': {
            'min': -0.5,
            'max': 1.0,
            'palette': ['ffffff', '7a8737', 'acbe4d', '0ae042', 'fff70b', 'ffaf38', 'ff641b']
        },
        'CIre': {
            'min': 0,
            'max': 3.0,
            'palette': ['8B4513', 'FFFF00', 'ADFF2F', '32CD32', '228B22', '006400']
        },
        'MSI': {
            'min': 0.0,
            'max': 2.5,
            'palette': ['006400', '228B22', '9ACD32', 'FFFF00', 'FFA500', 'FF0000']
        }
    }
    return params.get(index_name, params['NDVI'])


# -------------------------------------------------------
# ENDPOINT 1: NDVI/Multi-índice (ACTUALIZADO CON DATOS MEJORADOS)
# -------------------------------------------------------
@app.route("/api/ndvi", methods=["POST"])
def calculate_index():
    """
    Endpoint principal para calcular índices espectrales
    Body: {
        "geometry": {...},
        "date": "2025-10-03",
        "index": "NDVI"  // opcional, default NDVI
    }
    """
    try:
        data = request.json
        date_str = data.get("date")
        geojson_feature = data.get("geometry")
        index_name = data.get("index", "NDVI")

        if not date_str or not geojson_feature:
            return jsonify({
                "status": "error",
                "message": "Faltan parámetros: 'date' o 'geometry'."
            }), 400

        # Validar índice
        if index_name not in ['NDVI', 'NBR', 'CIre', 'MSI']:
            return jsonify({
                "status": "error",
                "message": f"Índice no soportado: {index_name}. Use: NDVI, NBR, CIre, MSI"
            }), 400

        # Crear geometría con nueva lógica (punto: 5m, polígonos: sin buffer)
        geometry_ee = build_geometry_aoi(geojson_feature)
        
        # Calcular áreas
        area_m2 = geometry_ee.area().getInfo()
        area_km2 = round(area_m2 / 1e6, 4)
        area_ha = round(area_m2 / 10000, 4)
        
        date_ee = ee.Date(date_str)
        
        # Colección con ventana de ±1 mes
        col = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")\
            .filterBounds(geometry_ee)\
            .filterDate(date_ee.advance(-1, 'month'), date_ee.advance(1, 'month'))\
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))\
            .map(mask_s2_clouds)\
            .map(lambda img: add_spectral_index(img, index_name))
        
        imgs_found = col.size().getInfo()
        
        if imgs_found == 0:
            return jsonify({
                "status": "warning",
                "message": "⚠️ No hay imágenes disponibles en el periodo.",
                "images_count": 0,
                "area_km2": area_km2,
                "area_ha": area_ha
            }), 200

        # Obtener imagen más cercana a la fecha
        def add_diff(img):
            diff = ee.Number(img.date().difference(date_ee, "day")).abs()
            return img.set("diff", diff)

        nearest = ee.Image(col.map(add_diff).sort("diff").first())
        
        # Estadísticas COMPLETAS del índice
        stats = nearest.select(index_name).reduceRegion(
            reducer=ee.Reducer.mean().combine(
                ee.Reducer.minMax(), '', True
            ).combine(
                ee.Reducer.stdDev(), '', True
            ).combine(
                ee.Reducer.median(), '', True
            ).combine(
                ee.Reducer.percentile([25, 75]), '', True
            ),
            geometry=geometry_ee,
            scale=10,
            maxPixels=1e13,
            bestEffort=True
        ).getInfo()
        
        mean_value = stats.get(f'{index_name}_mean')
        min_value = stats.get(f'{index_name}_min')
        max_value = stats.get(f'{index_name}_max')
        std_value = stats.get(f'{index_name}_stdDev')
        median_value = stats.get(f'{index_name}_median')
        p25_value = stats.get(f'{index_name}_p25')
        p75_value = stats.get(f'{index_name}_p75')
        
        # Visualización
        vis_params = get_visualization_params(index_name)
        clipped = nearest.select(index_name).clip(geometry_ee)
        tile_url = clipped.getMapId(vis_params)["tile_fetcher"].url_format
        
        # Bounds
        try:
            bounds_obj = geometry_ee.bounds().getInfo()
            coords = bounds_obj["coordinates"][0]
            lats = [c[1] for c in coords]
            lons = [c[0] for c in coords]
            bounds = [[min(lats), min(lons)], [max(lats), max(lons)]]
        except Exception:
            bounds = None

        # NUEVO: Información detallada de imágenes
        images_info = get_image_dates_info(col)
        image_date_used = nearest.date().format("YYYY-MM-dd").getInfo()
        
        # Nubosidad de la imagen usada
        cloud_percentage = nearest.get('CLOUDY_PIXEL_PERCENTAGE').getInfo()
        
        # Tile de Sentinel-2
        mgrs_tile = nearest.get('MGRS_TILE').getInfo()
        
        # Información del satélite
        spacecraft = nearest.get('SPACECRAFT_NAME').getInfo()

        # RESPUESTA MEJORADA
        return jsonify({
            "status": "success",
            "index": index_name,
            
            # ESTADÍSTICAS DEL ÍNDICE (mejoradas)
            "statistics": {
                "mean": round(mean_value, 4) if mean_value else None,
                "min": round(min_value, 4) if min_value else None,
                "max": round(max_value, 4) if max_value else None,
                "std": round(std_value, 4) if std_value else None,
                "median": round(median_value, 4) if median_value else None,
                "p25": round(p25_value, 4) if p25_value else None,
                "p75": round(p75_value, 4) if p75_value else None,
                "range": round(max_value - min_value, 4) if (max_value and min_value) else None
            },
            
            # Compatibilidad con código antiguo
            f"mean_{index_name.lower()}": round(mean_value, 4) if mean_value else None,
            "mean_ndvi": round(mean_value, 4) if mean_value else None,
            
            # INFORMACIÓN DE IMÁGENES (mejorada)
            "imagery": {
                "images_available": images_info,  # Lista completa con detalles
                "images_count": imgs_found,
                "image_used": {
                    "date": image_date_used,
                    "cloud_percentage": round(cloud_percentage, 2) if cloud_percentage else None,
                    "tile": mgrs_tile,
                    "satellite": spacecraft
                }
            },
            
            # Mantener para compatibilidad
            "image_date": image_date_used,
            "images_found": imgs_found,
            "cloud_percentage": round(cloud_percentage, 2) if cloud_percentage else None,
            
            # GEOMETRÍA
            "geometry": {
                "area_km2": area_km2,
                "area_ha": area_ha,
                "area_m2": round(area_m2, 2),
                "type": geojson_feature.get("geometry", {}).get("type", "Unknown")
            },
            
            # Mantener para compatibilidad
            "area_km2": area_km2,
            
            # VISUALIZACIÓN
            "tile_url": tile_url,
            "bounds": bounds,
            
            # NUEVO: URL para descarga futura (preparado para GeoTIFF)
            "download_url": None,  # Implementar en futuro
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


# -------------------------------------------------------
# ENDPOINT 2: Series Temporales con Tendencia (ACTUALIZADO)
# -------------------------------------------------------
@app.route("/api/timeseries/trend", methods=["POST"])
def timeseries_with_trend():
    """
    Calcula serie temporal mensual + pendiente de tendencia
    Body: {
        "geometry": {...},
        "index": "NDVI",
        "start_month": "2023-09",
        "end_month": "2025-10"
    }
    """
    try:
        data = request.json
        geometry = data.get("geometry")
        index_name = data.get("index", "NDVI")
        start_month = data.get("start_month")
        end_month = data.get("end_month")
        
        if not all([geometry, start_month, end_month]):
            return jsonify({
                "status": "error",
                "message": "Faltan parámetros requeridos"
            }), 400
        
        # Usar nueva función de geometría
        geometry_ee = build_geometry_aoi(geometry)
        area_m2 = geometry_ee.area().getInfo()
        area_km2 = round(area_m2 / 1e6, 4)
        area_ha = round(area_m2 / 10000, 4)
        
        # Parsear fechas
        start_parts = start_month.split('-')
        end_parts = end_month.split('-')
        start_date = ee.Date.fromYMD(
            ee.Number.parse(start_parts[0]),
            ee.Number.parse(start_parts[1]),
            1
        )
        end_date = ee.Date.fromYMD(
            ee.Number.parse(end_parts[0]),
            ee.Number.parse(end_parts[1]),
            1
        ).advance(1, 'month')
        
        # Colección base
        col = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")\
            .filterBounds(geometry_ee)\
            .filterDate(start_date, end_date)\
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))\
            .map(mask_s2_clouds)\
            .map(lambda img: add_spectral_index(img, index_name))\
            .select([index_name])
        
        imgs_count = col.size().getInfo()
        if imgs_count == 0:
            return jsonify({
                "status": "warning",
                "message": "No hay imágenes disponibles en el periodo",
                "images_found": 0,
                "area_km2": area_km2,
                "area_ha": area_ha
            }), 200
        
        # Serie mensual
        months_diff = end_date.difference(start_date, 'month').round()
        seq = ee.List.sequence(0, months_diff.subtract(1))
        
        def monthly_composite(i):
            start = start_date.advance(i, 'month')
            end = start.advance(1, 'month')
            img = col.filterDate(start, end).mean()
            return img.set('system:time_start', start.millis())
        
        monthly = ee.ImageCollection(seq.map(monthly_composite))
        
        # Calcular serie temporal
        def compute_monthly_stats(img):
            stats = img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry_ee,
                scale=10,
                maxPixels=1e13,
                bestEffort=True
            )
            date = ee.Date(img.get('system:time_start')).format('YYYY-MM')
            return ee.Feature(None, {
                'date': date,
                'mean': stats.get(index_name)
            })
        
        series = monthly.map(compute_monthly_stats).filter(
            ee.Filter.notNull(['mean'])
        )
        
        series_list = series.toList(series.size())
        series_size = series.size().getInfo()
        
        if series_size == 0:
            return jsonify({
                "status": "warning",
                "message": "No hay datos disponibles en el periodo seleccionado",
                "images_found": imgs_count,
                "area_km2": area_km2
            }), 200
        
        # Calcular deltas
        def compute_delta(i):
            i = ee.Number(i)
            curr = ee.Feature(series_list.get(i))
            prev_val = ee.Algorithms.If(
                i.eq(0),
                curr.get('mean'),
                ee.Feature(series_list.get(i.subtract(1))).get('mean')
            )
            curr_val = ee.Number(curr.get('mean'))
            delta = curr_val.subtract(ee.Number(prev_val))
            return curr.set('delta', delta)
        
        series_with_delta = ee.FeatureCollection(
            ee.List.sequence(0, series_size - 1).map(compute_delta)
        )
        
        # Calcular pendiente (trend)
        def add_time_band(img):
            t = ee.Date(img.get('system:time_start')).difference(start_date, 'year')
            return img.addBands(
                ee.Image.constant(t).toFloat().rename('t')
            )
        
        monthly_with_time = monthly.map(add_time_band)
        fit = monthly_with_time.select([index_name, 't']).reduce(
            ee.Reducer.linearFit()
        )
        slope = fit.select('scale')
        
        # URL del mapa de pendiente
        slope_vis = {
            'min': -0.05,
            'max': 0.05,
            'palette': ['#dc2626', '#f59e0b', '#ffffff', '#a3e635', '#047857']
        }
        slope_clipped = slope.clip(geometry_ee)
        slope_tile_url = slope_clipped.getMapId(slope_vis)['tile_fetcher'].url_format
        
        # Estadísticas de pendiente
        slope_stats = slope_clipped.reduceRegion(
            reducer=ee.Reducer.minMax().combine(
                ee.Reducer.mean(), '', True
            ),
            geometry=geometry_ee,
            scale=10,
            maxPixels=1e13,
            bestEffort=True
        ).getInfo()
        
        # Obtener datos de la serie
        timeseries_data = series_with_delta.getInfo()['features']
        
        return jsonify({
            "status": "success",
            "index": index_name,
            "images_found": imgs_count,
            "geometry": {
                "area_km2": area_km2,
                "area_ha": area_ha
            },
            "area_km2": area_km2,  # Compatibilidad
            "timeseries": [
                {
                    'date': f['properties']['date'],
                    'mean': round(f['properties']['mean'], 4) if f['properties']['mean'] else None,
                    'delta': round(f['properties']['delta'], 4) if f['properties']['delta'] else None
                }
                for f in timeseries_data
            ],
            "slope_tile_url": slope_tile_url,
            "slope_stats": {
                'min': slope_stats.get('scale_min'),
                'max': slope_stats.get('scale_max'),
                'mean': slope_stats.get('scale_mean')
            }
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


# -------------------------------------------------------
# ENDPOINT 3: Análisis con Umbrales (ACTUALIZADO)
# -------------------------------------------------------
@app.route("/api/analysis/thresholds", methods=["POST"])
def analysis_with_thresholds():
    """
    Análisis temporal con clasificación por umbrales
    Body: {
        "geometry": {...},
        "index": "NDVI",
        "start_month": "2023-09",
        "end_month": "2025-10"
    }
    """
    try:
        data = request.json
        geometry = data.get("geometry")
        index_name = data.get("index", "NDVI")
        start_month = data.get("start_month")
        end_month = data.get("end_month")
        
        # Umbrales por índice
        thresholds = {
            'NDVI': {'sin_afeccion': 0.30, 'advertencia': 0.24, 'alerta': 0.20},
            'NBR': {'sin_afeccion': 0.44, 'advertencia': 0.27, 'alerta': 0.10},
            'CIre': {'sin_afeccion': 0.40, 'advertencia': 0.30, 'alerta': 0.20},
            'MSI': {'sin_afeccion': 1.00, 'advertencia': 1.30, 'alerta': 1.60}
        }
        
        geometry_ee = build_geometry_aoi(geometry)
        area_m2 = geometry_ee.area().getInfo()
        area_km2 = round(area_m2 / 1e6, 4)
        area_ha = round(area_m2 / 10000, 4)
        
        # Parsear fechas
        start_parts = start_month.split('-')
        end_parts = end_month.split('-')
        start_date = ee.Date.fromYMD(
            ee.Number.parse(start_parts[0]),
            ee.Number.parse(start_parts[1]),
            1
        )
        end_date = ee.Date.fromYMD(
            ee.Number.parse(end_parts[0]),
            ee.Number.parse(end_parts[1]),
            1
        ).advance(1, 'month')
        
        # Colección
        col = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")\
            .filterBounds(geometry_ee)\
            .filterDate(start_date, end_date)\
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))\
            .map(mask_s2_clouds)\
            .map(lambda img: add_spectral_index(img, index_name))\
            .select([index_name])
        
        imgs_count = col.size().getInfo()
        if imgs_count == 0:
            return jsonify({
                "status": "warning",
                "message": "No hay imágenes disponibles",
                "area_km2": area_km2,
                "area_ha": area_ha
            }), 200
        
        # Serie mensual
        months_diff = end_date.difference(start_date, 'month').round()
        seq = ee.List.sequence(0, months_diff.subtract(1))
        
        def monthly_composite(i):
            start = start_date.advance(i, 'month')
            end = start.advance(1, 'month')
            img = col.filterDate(start, end).mean()
            return img.set('system:time_start', start.millis())
        
        monthly = ee.ImageCollection(seq.map(monthly_composite))
        
        def compute_stats(img):
            stats = img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry_ee,
                scale=10,
                maxPixels=1e13,
                bestEffort=True
            )
            date = ee.Date(img.get('system:time_start')).format('YYYY-MM')
            return ee.Feature(None, {
                'date': date,
                'mean': stats.get(index_name)
            })
        
        series = monthly.map(compute_stats).filter(
            ee.Filter.notNull(['mean'])
        )
        
        timeseries_data = series.getInfo()['features']
        
        # Clasificar según umbrales
        alerts = []
        for point in timeseries_data:
            value = point['properties']['mean']
            date = point['properties']['date']
            
            if value is None:
                continue
            
            # MSI es inverso (mayor = peor)
            if index_name == 'MSI':
                if value > thresholds[index_name]['alerta']:
                    alerts.append({
                        'date': date,
                        'level': 'alerta',
                        'value': round(value, 4)
                    })
                elif value > thresholds[index_name]['advertencia']:
                    alerts.append({
                        'date': date,
                        'level': 'advertencia',
                        'value': round(value, 4)
                    })
            else:
                # Resto de índices: menor = peor
                if value < thresholds[index_name]['alerta']:
                    alerts.append({
                        'date': date,
                        'level': 'alerta',
                        'value': round(value, 4)
                    })
                elif value < thresholds[index_name]['advertencia']:
                    alerts.append({
                        'date': date,
                        'level': 'advertencia',
                        'value': round(value, 4)
                    })
        
        return jsonify({
            "status": "success",
            "index": index_name,
            "images_found": imgs_count,
            "geometry": {
                "area_km2": area_km2,
                "area_ha": area_ha
            },
            "area_km2": area_km2,  # Compatibilidad
            "timeseries": [
                {
                    'date': f['properties']['date'],
                    'mean': round(f['properties']['mean'], 4) if f['properties']['mean'] else None
                }
                for f in timeseries_data
            ],
            "thresholds": thresholds[index_name],
            "alerts": alerts
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


# -------------------------------------------------------
# ENDPOINT 4: Compuesto Temporal (ACTUALIZADO)
# -------------------------------------------------------
@app.route("/api/composite/temporal", methods=["POST"])
def composite_temporal():
    """
    Crea imagen compuesta con ventana temporal
    Body: {
        "geometry": {...},
        "date": "2025-03-27",
        "days_window": 7,
        "max_cloud": 40,
        "indices": ["NDVI", "NBR"]
    }
    """
    try:
        data = request.json
        geometry = data.get("geometry")
        center_date_str = data.get("date")
        days_window = data.get("days_window", 7)
        max_cloud = data.get("max_cloud", 40)
        indices = data.get("indices", ["NDVI"])
        
        geometry_ee = build_geometry_aoi(geometry)
        area_m2 = geometry_ee.area().getInfo()
        area_km2 = round(area_m2 / 1e6, 4)
        area_ha = round(area_m2 / 10000, 4)
        center_date = ee.Date(center_date_str)
        
        start_date = center_date.advance(-days_window, 'day')
        end_date = center_date.advance(days_window, 'day')
        
        # Colección
        col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')\
            .filterBounds(geometry_ee)\
            .filterDate(start_date, end_date)\
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud))\
            .map(mask_s2_clouds)\
            .map(add_all_indices)
        
        imgs_count = col.size().getInfo()
        if imgs_count == 0:
            return jsonify({
                "status": "warning",
                "message": "No hay imágenes en la ventana temporal",
                "area_km2": area_km2,
                "area_ha": area_ha
            }), 200
        
        # Imagen mediana
        median_img = col.median().clip(geometry_ee)
        
        # Información de imágenes usadas
        images_info = get_image_dates_info(col)
        
        # Generar tiles y calcular estadísticas
        tiles = {}
        stats = {}
        
        for idx in indices:
            # Visualización
            vis_params = get_visualization_params(idx)
            
            # Tile URL
            tile_url = median_img.select(idx).getMapId(vis_params)['tile_fetcher'].url_format
            tiles[idx] = tile_url
            
            # Estadísticas completas
            idx_stats = median_img.select(idx).reduceRegion(
                reducer=ee.Reducer.mean().combine(
                    ee.Reducer.minMax(), '', True
                ).combine(
                    ee.Reducer.stdDev(), '', True
                ),
                geometry=geometry_ee,
                scale=10,
                maxPixels=1e13,
                bestEffort=True
            ).getInfo()
            
            stats[idx] = {
                'mean': round(idx_stats.get(f'{idx}_mean'), 4) if idx_stats.get(f'{idx}_mean') else None,
                'min': round(idx_stats.get(f'{idx}_min'), 4) if idx_stats.get(f'{idx}_min') else None,
                'max': round(idx_stats.get(f'{idx}_max'), 4) if idx_stats.get(f'{idx}_max') else None,
                'std': round(idx_stats.get(f'{idx}_stdDev'), 4) if idx_stats.get(f'{idx}_stdDev') else None
            }
        
        return jsonify({
            "status": "success",
            "tiles": tiles,
            "statistics": stats,
            "imagery": {
                "images_used": imgs_count,
                "images_info": images_info,
                "date_range": {
                    'start': start_date.format('YYYY-MM-dd').getInfo(),
                    'end': end_date.format('YYYY-MM-dd').getInfo()
                }
            },
            "geometry": {
                "area_km2": area_km2,
                "area_ha": area_ha
            },
            # Compatibilidad
            "images_used": imgs_count,
            "stats": stats
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


# -------------------------------------------------------
# ENDPOINT 5: Comparación Multi-índice (ACTUALIZADO)
# -------------------------------------------------------
@app.route("/api/indices/compare", methods=["POST"])
def compare_indices():
    """
    Calcula múltiples índices para la misma fecha/geometría
    Body: {
        "geometry": {...},
        "date": "2025-03-27",
        "indices": ["NDVI", "NBR", "CIre", "MSI"]
    }
    """
    try:
        data = request.json
        geometry = data.get("geometry")
        date_str = data.get("date")
        indices = data.get("indices", ["NDVI", "NBR", "CIre", "MSI"])
        
        geometry_ee = build_geometry_aoi(geometry)
        area_m2 = geometry_ee.area().getInfo()
        area_km2 = round(area_m2 / 1e6, 4)
        area_ha = round(area_m2 / 10000, 4)
        date_ee = ee.Date(date_str)
        
        # Colección
        col = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")\
            .filterBounds(geometry_ee)\
            .filterDate(date_ee.advance(-1, 'month'), date_ee.advance(1, 'month'))\
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))\
            .map(mask_s2_clouds)\
            .map(add_all_indices)
        
        imgs_found = col.size().getInfo()
        if imgs_found == 0:
            return jsonify({
                "status": "warning",
                "message": "No hay imágenes disponibles",
                "area_km2": area_km2,
                "area_ha": area_ha
            }), 200
        
        # Imagen más cercana
        def add_diff(img):
            diff = ee.Number(img.date().difference(date_ee, "day")).abs()
            return img.set("diff", diff)
        
        nearest = ee.Image(col.map(add_diff).sort("diff").first())
        image_date = nearest.date().format("YYYY-MM-dd").getInfo()
        cloud_percentage = nearest.get('CLOUDY_PIXEL_PERCENTAGE').getInfo()
        
        results = {}
        for idx in indices:
            # Estadísticas completas
            idx_stats = nearest.select(idx).reduceRegion(
                reducer=ee.Reducer.mean().combine(
                    ee.Reducer.minMax(), '', True
                ).combine(
                    ee.Reducer.stdDev(), '', True
                ),
                geometry=geometry_ee,
                scale=10,
                maxPixels=1e13,
                bestEffort=True
            ).getInfo()
            
            # Visualización
            vis_params = get_visualization_params(idx)
            clipped = nearest.select(idx).clip(geometry_ee)
            tile_url = clipped.getMapId(vis_params)['tile_fetcher'].url_format
            
            results[idx] = {
                'mean': round(idx_stats.get(f'{idx}_mean'), 4) if idx_stats.get(f'{idx}_mean') else None,
                'min': round(idx_stats.get(f'{idx}_min'), 4) if idx_stats.get(f'{idx}_min') else None,
                'max': round(idx_stats.get(f'{idx}_max'), 4) if idx_stats.get(f'{idx}_max') else None,
                'std': round(idx_stats.get(f'{idx}_stdDev'), 4) if idx_stats.get(f'{idx}_stdDev') else None,
                'tile_url': tile_url
            }
        
        return jsonify({
            "status": "success",
            "imagery": {
                "image_date": image_date,
                "images_found": imgs_found,
                "cloud_percentage": round(cloud_percentage, 2) if cloud_percentage else None
            },
            "geometry": {
                "area_km2": area_km2,
                "area_ha": area_ha
            },
            "results": results,
            # Compatibilidad
            "image_date": image_date,
            "images_found": imgs_found
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


# -------------------------------------------------------
# ENDPOINT 6: Detección de Anomalías (ACTUALIZADO)
# -------------------------------------------------------
@app.route("/api/anomalies/detect", methods=["POST"])
def detect_anomalies():
    """
    Detecta anomalías basándose en desviación estándar histórica
    Body: {
        "geometry": {...},
        "index": "NDVI",
        "reference_start": "2020-01",
        "reference_end": "2023-12",
        "test_date": "2024-08-15"
    }
    """
    try:
        data = request.json
        geometry = data.get("geometry")
        index_name = data.get("index", "NDVI")
        ref_start = data.get("reference_start")
        ref_end = data.get("reference_end")
        test_date = data.get("test_date")
        
        geometry_ee = build_geometry_aoi(geometry)
        area_m2 = geometry_ee.area().getInfo()
        area_km2 = round(area_m2 / 1e6, 4)
        area_ha = round(area_m2 / 10000, 4)
        
        # Periodo de referencia (histórico)
        ref_start_parts = ref_start.split('-')
        ref_end_parts = ref_end.split('-')
        ref_start_date = ee.Date.fromYMD(
            ee.Number.parse(ref_start_parts[0]),
            ee.Number.parse(ref_start_parts[1]),
            1
        )
        ref_end_date = ee.Date.fromYMD(
            ee.Number.parse(ref_end_parts[0]),
            ee.Number.parse(ref_end_parts[1]),
            1
        ).advance(1, 'month')
        
        # Colección histórica
        historical_col = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")\
            .filterBounds(geometry_ee)\
            .filterDate(ref_start_date, ref_end_date)\
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))\
            .map(mask_s2_clouds)\
            .map(lambda img: add_spectral_index(img, index_name))\
            .select([index_name])
        
        hist_count = historical_col.size().getInfo()
        if hist_count == 0:
            return jsonify({
                "status": "warning",
                "message": "No hay imágenes en el periodo de referencia",
                "area_km2": area_km2,
                "area_ha": area_ha
            }), 200
        
        # Calcular media y desviación estándar histórica
        historical_mean_img = historical_col.mean()
        historical_std_img = historical_col.reduce(ee.Reducer.stdDev())
        
        historical_stats = historical_mean_img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry_ee,
            scale=10,
            maxPixels=1e13,
            bestEffort=True
        )
        historical_mean = historical_stats.get(index_name).getInfo()
        
        historical_std_stats = historical_std_img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry_ee,
            scale=10,
            maxPixels=1e13,
            bestEffort=True
        )
        historical_std = historical_std_stats.get(f'{index_name}_stdDev').getInfo()
        
        # Valor actual (test)
        test_date_ee = ee.Date(test_date)
        test_col = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")\
            .filterBounds(geometry_ee)\
            .filterDate(test_date_ee.advance(-15, 'day'), test_date_ee.advance(15, 'day'))\
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))\
            .map(mask_s2_clouds)\
            .map(lambda img: add_spectral_index(img, index_name))\
            .select([index_name])
        
        test_count = test_col.size().getInfo()
        if test_count == 0:
            return jsonify({
                "status": "warning",
                "message": "No hay imágenes en la fecha de test",
                "area_km2": area_km2,
                "area_ha": area_ha
            }), 200
        
        def add_diff(img):
            diff = ee.Number(img.date().difference(test_date_ee, "day")).abs()
            return img.set("diff", diff)
        
        test_img = ee.Image(test_col.map(add_diff).sort("diff").first())
        
        test_stats = test_img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry_ee,
            scale=10,
            maxPixels=1e13,
            bestEffort=True
        )
        current_value = test_stats.get(index_name).getInfo()
        
        # Calcular Z-score
        if historical_std and historical_std > 0:
            z_score = (current_value - historical_mean) / historical_std
        else:
            z_score = 0
        
        # Clasificar severidad
        abs_z = abs(z_score)
        if abs_z > 2.5:
            severity = "high"
            anomaly_detected = True
        elif abs_z > 1.5:
            severity = "medium"
            anomaly_detected = True
        elif abs_z > 1.0:
            severity = "low"
            anomaly_detected = True
        else:
            severity = "none"
            anomaly_detected = False
        
        # Determinar tipo de anomalía
        if z_score < -1.5:
            anomaly_type = "degradation"
        elif z_score > 1.5:
            anomaly_type = "improvement"
        else:
            anomaly_type = "normal"
        
        return jsonify({
            "status": "success",
            "anomaly_detected": anomaly_detected,
            "anomaly_type": anomaly_type,
            "severity": severity,
            "z_score": round(z_score, 3),
            "historical_mean": round(historical_mean, 4),
            "historical_std": round(historical_std, 4),
            "current_value": round(current_value, 4),
            "reference_images": hist_count,
            "test_images": test_count,
            "test_image_date": test_img.date().format("YYYY-MM-dd").getInfo(),
            "geometry": {
                "area_km2": area_km2,
                "area_ha": area_ha
            }
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


# -------------------------------------------------------
# ENDPOINT 7: Health Check
# -------------------------------------------------------
@app.route("/api/health", methods=["GET"])
def health_check():
    """Verifica el estado del servicio"""
    try:
        test = ee.Number(1).getInfo()
        return jsonify({
            "status": "healthy",
            "service": "GeoVisor Backend API",
            "gee_initialized": True,
            "version": "2.1.0"
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "service": "GeoVisor Backend API",
            "gee_initialized": False,
            "error": str(e)
        }), 500


# -------------------------------------------------------
# ENDPOINT 8: Listar Índices Disponibles
# -------------------------------------------------------
@app.route("/api/indices/list", methods=["GET"])
def list_indices():
    """Retorna información sobre los índices espectrales disponibles"""
    indices_info = {
        "NDVI": {
            "name": "Normalized Difference Vegetation Index",
            "description": "Mide el vigor vegetal a partir de NIR y Red",
            "formula": "(NIR - Red) / (NIR + Red)",
            "range": [-1, 1],
            "thresholds": {
                "sin_afeccion": 0.30,
                "advertencia": 0.24,
                "alerta": 0.20
            },
            "interpretation": {
                "high": "> 0.30 - Vegetación saludable",
                "medium": "0.24-0.30 - Vigilancia",
                "low": "0.20-0.24 - Advertencia",
                "critical": "< 0.20 - Alerta"
            }
        },
        "NBR": {
            "name": "Normalized Burn Ratio",
            "description": "Detecta áreas quemadas o perturbaciones severas",
            "formula": "(NIR - SWIR) / (NIR + SWIR)",
            "range": [-1, 1],
            "thresholds": {
                "sin_afeccion": 0.44,
                "advertencia": 0.27,
                "alerta": 0.10
            },
            "interpretation": {
                "high": "> 0.44 - Vegetación buena",
                "medium": "0.27-0.44 - Aceptable",
                "low": "0.10-0.27 - Estrés leve",
                "critical": "< 0.10 - Daño severo"
            }
        },
        "CIre": {
            "name": "Chlorophyll Index RedEdge",
            "description": "Estima contenido de clorofila",
            "formula": "(NIR / RedEdge) - 1",
            "range": [0, 5],
            "thresholds": {
                "sin_afeccion": 0.40,
                "advertencia": 0.30,
                "alerta": 0.20
            },
            "interpretation": {
                "high": "> 0.40 - Alta clorofila",
                "medium": "0.30-0.40 - Media",
                "low": "0.20-0.30 - Baja",
                "critical": "< 0.20 - Muy baja"
            }
        },
        "MSI": {
            "name": "Moisture Stress Index",
            "description": "Indica estrés hídrico (valores altos = más estrés)",
            "formula": "SWIR / NIR",
            "range": [0, 3],
            "thresholds": {
                "sin_afeccion": 1.00,
                "advertencia": 1.30,
                "alerta": 1.60
            },
            "interpretation": {
                "high": "< 1.00 - Buena humedad",
                "medium": "1.00-1.30 - Estrés leve",
                "low": "1.30-1.60 - Estrés moderado",
                "critical": "> 1.60 - Estrés severo"
            },
            "inverted": True
        }
    }
    
    return jsonify({
        "status": "success",
        "indices": indices_info
    })

# -------------------------------------------------------
# ENDPOINT 9: Descarga GeoTIFF (VERSIÓN COMPLETA)
# -------------------------------------------------------
@app.route("/api/download/geotiff", methods=["POST"])
def download_geotiff():
    """
    Genera URL de descarga de GeoTIFF
    Body: {
        "geometry": {...},
        "date": "2025-10-03",
        "index": "NDVI"
    }
    """
    print("🔵 Endpoint /api/download/geotiff llamado")  # DEBUG
    
    try:
        data = request.json
        print(f"📥 Datos recibidos: {data}")  # DEBUG
        
        geometry = data.get("geometry")
        date_str = data.get("date")
        index_name = data.get("index", "NDVI")
        
        if not all([geometry, date_str]):
            return jsonify({
                "status": "error",
                "message": "Faltan parámetros requeridos: geometry y date"
            }), 400
        
        # Validar índice
        if index_name not in ['NDVI', 'NBR', 'CIre', 'MSI']:
            return jsonify({
                "status": "error",
                "message": f"Índice no soportado: {index_name}"
            }), 400
        
        # Crear geometría
        geometry_ee = build_geometry_aoi(geometry)
        
        # Calcular área y validar límites
        area_m2 = geometry_ee.area().getInfo()
        area_km2 = round(area_m2 / 1e6, 4)
        
        print(f"📐 Área calculada: {area_km2} km²")  # DEBUG
        
        # VALIDACIÓN DE ÁREA
        if area_km2 > 100:
            return jsonify({
                "status": "error",
                "message": f"El área es demasiado grande ({area_km2:.2f} km²). Límite: 100 km².",
                "area_km2": area_km2
            }), 400
        
        if area_km2 > 50:
            print(f"⚠️ Área grande: {area_km2:.2f} km²")
        
        date_ee = ee.Date(date_str)
        
        # Colección
        col = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")\
            .filterBounds(geometry_ee)\
            .filterDate(date_ee.advance(-1, 'month'), date_ee.advance(1, 'month'))\
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))\
            .map(mask_s2_clouds)\
            .map(lambda img: add_spectral_index(img, index_name))
        
        imgs_count = col.size().getInfo()
        print(f"🛰️ Imágenes encontradas: {imgs_count}")  # DEBUG
        
        if imgs_count == 0:
            return jsonify({
                "status": "error",
                "message": "No hay imágenes disponibles para generar el GeoTIFF"
            }), 404
        
        # Imagen más cercana
        def add_diff(img):
            diff = ee.Number(img.date().difference(date_ee, "day")).abs()
            return img.set("diff", diff)
        
        nearest = ee.Image(col.map(add_diff).sort("diff").first())
        clipped = nearest.select(index_name).clip(geometry_ee)
        
        # Obtener fecha de la imagen
        image_date = nearest.date().format("YYYY-MM-dd").getInfo()
        
        print(f"✅ Generando GeoTIFF: {index_name} - {image_date} - {area_km2} km²")
        
        # Generar URL de descarga
        try:
            download_url = clipped.getDownloadURL({
                'scale': 10,
                'crs': 'EPSG:4326',
                'fileFormat': 'GeoTIFF',
                'region': geometry_ee.bounds().getInfo()['coordinates']
            })
            print(f"🔗 URL generada exitosamente")  # DEBUG
        except Exception as download_error:
            print(f"❌ Error generando URL: {download_error}")
            traceback.print_exc()
            return jsonify({
                "status": "error",
                "message": f"Error al generar URL de descarga: {str(download_error)}"
            }), 500
        
        return jsonify({
            "status": "success",
            "download_url": download_url,
            "filename": f"{index_name}_{image_date}.tif",
            "image_date": image_date,
            "index": index_name,
            "images_found": imgs_count,
            "area_km2": area_km2,
            "message": f"GeoTIFF generado correctamente ({area_km2:.2f} km²). Válido 24h."
        })
        
    except Exception as e:
        print(f"❌ Error en download_geotiff: {e}")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": f"Error al generar GeoTIFF: {str(e)}"
        }), 500


# -------------------------------------------------------
# ENDPOINT 10: Calculadora de Umbrales Inteligente (CORREGIDO)
# -------------------------------------------------------
@app.route("/api/thresholds/calculate", methods=["POST"])
def calculate_thresholds():
    """
    Calcula umbrales óptimos basándose en análisis estadístico de datos históricos.
    """
    print("🔵 Endpoint /api/thresholds/calculate llamado")
    
    try:
        data = request.json
        print(f"📥 Datos recibidos: {data}")
        
        geometry = data.get('geometry')
        index_name = data.get('index', 'NDVI')
        start_month = data.get('start_month', '2017-04')
        end_month = data.get('end_month')
        method = data.get('method', 'percentiles')
        
        if not geometry:
            return jsonify({'status': 'error', 'message': 'Geometría requerida'}), 400
        
        if index_name not in ['NDVI', 'NBR', 'CIre', 'MSI']:
            return jsonify({'status': 'error', 'message': f'Índice no soportado: {index_name}'}), 400
        
        geometry_ee = build_geometry_aoi(geometry)
        area_m2 = geometry_ee.area().getInfo()
        area_km2 = round(area_m2 / 1e6, 4)
        print(f"📐 Área: {area_km2} km²")
        
        # Parsear fechas
        start_parts = start_month.split('-')
        end_parts = end_month.split('-')
        start_date_ee = ee.Date.fromYMD(
            ee.Number.parse(start_parts[0]),
            ee.Number.parse(start_parts[1]),
            1
        )
        end_date_ee = ee.Date.fromYMD(
            ee.Number.parse(end_parts[0]),
            ee.Number.parse(end_parts[1]),
            1
        ).advance(1, 'month')
        
        print(f"📅 Periodo: {start_month} a {end_month}")
        
        # Colección base
        collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(geometry_ee) \
            .filterDate(start_date_ee, end_date_ee) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)) \
            .map(mask_s2_clouds) \
            .map(lambda img: add_spectral_index(img, index_name)) \
            .select([index_name])
        
        total_images = collection.size().getInfo()
        print(f"🛰️ Imágenes totales: {total_images}")
        
        if total_images == 0:
            return jsonify({
                'status': 'error',
                'message': 'No se encontraron imágenes para el periodo seleccionado'
            }), 404
        
        # Calcular composites mensuales
        months_diff = end_date_ee.difference(start_date_ee, 'month').round()
        seq = ee.List.sequence(0, months_diff.subtract(1))
        
        def monthly_composite(i):
            start = start_date_ee.advance(i, 'month')
            end = start.advance(1, 'month')
            monthly_col = collection.filterDate(start, end)
            
            # CRÍTICO: Solo crear imagen si hay datos
            count = monthly_col.size()
            
            # Crear imagen con banda condicional
            img = ee.Algorithms.If(
                count.gt(0),
                monthly_col.mean().set({
                    'system:time_start': start.millis(),
                    'count': count,
                    'has_data': True
                }),
                ee.Image.constant(0).rename(index_name).set({
                    'system:time_start': start.millis(),
                    'count': 0,
                    'has_data': False
                })
            )
            
            return ee.Image(img)
        
        monthly_collection = ee.ImageCollection(seq.map(monthly_composite))
        
        # Filtrar solo meses con datos
        monthly_with_data = monthly_collection.filter(ee.Filter.eq('has_data', True))
        
        # Calcular estadísticas mensuales
        def compute_monthly_stats(img):
            stats = img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry_ee,
                scale=10,
                maxPixels=1e9,
                bestEffort=True
            )
            date = ee.Date(img.get('system:time_start')).format('YYYY-MM')
            return ee.Feature(None, {
                'date': date,
                'mean': stats.get(index_name),
                'count': img.get('count')
            })
        
        series = monthly_with_data.map(compute_monthly_stats).filter(
            ee.Filter.notNull(['mean'])
        )
        
        print("⏳ Procesando serie temporal completa en GEE...")
        timeseries_data = series.getInfo()['features']
        
        timeseries = [
            {
                'date': f['properties']['date'],
                'mean': f['properties']['mean'],
                'count': f['properties']['count']
            }
            for f in timeseries_data
            if f['properties']['mean'] is not None
        ]
        
        print(f"✅ Serie temporal: {len(timeseries)} puntos válidos")
        
        if len(timeseries) == 0:
            return jsonify({
                'status': 'error',
                'message': 'No se encontraron datos válidos para el periodo seleccionado'
            }), 404
        
        # Extraer valores para análisis
        import numpy as np
        values = np.array([item['mean'] for item in timeseries])
        
        # Estadísticas básicas
        statistics = {
            'mean': float(np.mean(values)),
            'median': float(np.median(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'count': len(values),
            'p10': float(np.percentile(values, 10)),
            'p25': float(np.percentile(values, 25)),
            'p50': float(np.percentile(values, 50)),
            'p75': float(np.percentile(values, 75)),
            'p90': float(np.percentile(values, 90)),
        }
        
        print(f"📈 Estadísticas - Media: {statistics['mean']:.4f}, Std: {statistics['std']:.4f}")
        
        # Calcular umbrales según método
        if method == 'percentiles':
            suggested_thresholds = {
                'sin_afeccion': float(np.percentile(values, 75)),
                'advertencia': float(np.percentile(values, 50)),
                'alerta': float(np.percentile(values, 25))
            }
        elif method == 'std_deviation':
            mean = np.mean(values)
            std = np.std(values)
            suggested_thresholds = {
                'sin_afeccion': float(mean + 0.5 * std),
                'advertencia': float(mean),
                'alerta': float(mean - std)
            }
        elif method == 'seasonal':
            monthly_data = {}
            for item in timeseries:
                month = item['date'].split('-')[1]
                if month not in monthly_data:
                    monthly_data[month] = []
                monthly_data[month].append(item['mean'])
            
            all_p25 = []
            all_p50 = []
            all_p75 = []
            
            for month_key, month_values in monthly_data.items():
                if len(month_values) > 0:
                    all_p25.append(np.percentile(month_values, 25))
                    all_p50.append(np.percentile(month_values, 50))
                    all_p75.append(np.percentile(month_values, 75))
            
            suggested_thresholds = {
                'sin_afeccion': float(np.mean(all_p75)) if all_p75 else 0.6,
                'advertencia': float(np.mean(all_p50)) if all_p50 else 0.4,
                'alerta': float(np.mean(all_p25)) if all_p25 else 0.3
            }
        else:
            suggested_thresholds = {
                'sin_afeccion': float(np.percentile(values, 75)),
                'advertencia': float(np.percentile(values, 50)),
                'alerta': float(np.percentile(values, 25))
            }
        
        print(f"🎯 Umbrales calculados ({method}): {suggested_thresholds}")
        
        # Crear histograma
        hist, bin_edges = np.histogram(values, bins=20)
        histogram = {
            'frequencies': hist.tolist(),
            'bins': bin_edges[:-1].tolist(),
        }
        
        return jsonify({
            'status': 'success',
            'index': index_name,
            'method': method,
            'suggested_thresholds': suggested_thresholds,
            'statistics': statistics,
            'timeseries': timeseries,
            'histogram': histogram,
            'images_found': total_images,
            'period': {
                'start': start_month,
                'end': end_month
            },
            'area_km2': area_km2
        })
        
    except Exception as e:
        print(f"❌ Error en calculate_thresholds: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': f'Error al calcular umbrales: {str(e)}'
        }), 500


# -------------------------------------------------------
# ENDPOINT 11: Mapa de Diferencias/Cambios
# -------------------------------------------------------
@app.route("/api/analysis/change-map", methods=["POST"])
def generate_change_map():
    """
    Genera un mapa raster mostrando las mayores diferencias entre dos periodos
    Body: {
        "geometry": {...},
        "index": "NDVI",
        "baseline_start": "2020-01",
        "baseline_end": "2020-12",
        "comparison_start": "2024-01",
        "comparison_end": "2024-12"
    }
    """
    print("🔵 Endpoint /api/analysis/change-map llamado")
    
    try:
        data = request.json
        geometry = data.get('geometry')
        index_name = data.get('index', 'NDVI')
        baseline_start = data.get('baseline_start')
        baseline_end = data.get('baseline_end')
        comparison_start = data.get('comparison_start')
        comparison_end = data.get('comparison_end')
        
        if not all([geometry, baseline_start, baseline_end, comparison_start, comparison_end]):
            return jsonify({
                'status': 'error',
                'message': 'Faltan parámetros requeridos'
            }), 400
        
        geometry_ee = build_geometry_aoi(geometry)
        area_m2 = geometry_ee.area().getInfo()
        area_km2 = round(area_m2 / 1e6, 4)
        
        # Parsear fechas baseline
        baseline_start_parts = baseline_start.split('-')
        baseline_end_parts = baseline_end.split('-')
        baseline_start_date = ee.Date.fromYMD(
            ee.Number.parse(baseline_start_parts[0]),
            ee.Number.parse(baseline_start_parts[1]),
            1
        )
        baseline_end_date = ee.Date.fromYMD(
            ee.Number.parse(baseline_end_parts[0]),
            ee.Number.parse(baseline_end_parts[1]),
            1
        ).advance(1, 'month')
        
        # Parsear fechas comparison
        comparison_start_parts = comparison_start.split('-')
        comparison_end_parts = comparison_end.split('-')
        comparison_start_date = ee.Date.fromYMD(
            ee.Number.parse(comparison_start_parts[0]),
            ee.Number.parse(comparison_start_parts[1]),
            1
        )
        comparison_end_date = ee.Date.fromYMD(
            ee.Number.parse(comparison_end_parts[0]),
            ee.Number.parse(comparison_end_parts[1]),
            1
        ).advance(1, 'month')
        
        # Colección baseline
        baseline_col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(geometry_ee) \
            .filterDate(baseline_start_date, baseline_end_date) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)) \
            .map(mask_s2_clouds) \
            .map(lambda img: add_spectral_index(img, index_name)) \
            .select([index_name])
        
        baseline_count = baseline_col.size().getInfo()
        
        # Colección comparison
        comparison_col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(geometry_ee) \
            .filterDate(comparison_start_date, comparison_end_date) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)) \
            .map(mask_s2_clouds) \
            .map(lambda img: add_spectral_index(img, index_name)) \
            .select([index_name])
        
        comparison_count = comparison_col.size().getInfo()
        
        if baseline_count == 0 or comparison_count == 0:
            return jsonify({
                'status': 'error',
                'message': f'No hay suficientes imágenes. Baseline: {baseline_count}, Comparación: {comparison_count}'
            }), 404
        
        print(f"📊 Baseline: {baseline_count} imágenes, Comparación: {comparison_count} imágenes")
        
        # Calcular medias
        baseline_mean = baseline_col.mean()
        comparison_mean = comparison_col.mean()
        
        # Calcular diferencia (comparison - baseline)
        difference = comparison_mean.subtract(baseline_mean).rename('difference')
        
        # Calcular diferencia porcentual
        percent_change = difference.divide(baseline_mean).multiply(100).rename('percent_change')
        
        # Recortar a la geometría
        difference_clipped = difference.clip(geometry_ee)
        percent_clipped = percent_change.clip(geometry_ee)
        
        # Estadísticas de la diferencia
        diff_stats = difference_clipped.reduceRegion(
            reducer=ee.Reducer.mean().combine(
                ee.Reducer.minMax(), '', True
            ).combine(
                ee.Reducer.stdDev(), '', True
            ).combine(
                ee.Reducer.percentile([10, 25, 50, 75, 90]), '', True
            ),
            geometry=geometry_ee,
            scale=10,
            maxPixels=1e9,
            bestEffort=True
        ).getInfo()
        
        # Visualización del mapa de diferencias
        # Rojo = degradación, Blanco = sin cambio, Verde = mejora
        diff_viz = {
            'min': -0.3,
            'max': 0.3,
            'palette': [
                '#8B0000',  # Rojo oscuro (gran degradación)
                '#DC143C',  # Rojo
                '#FF6B6B',  # Rojo claro
                '#FFA07A',  # Naranja claro
                '#FFFFFF',  # Blanco (sin cambio)
                '#90EE90',  # Verde claro
                '#32CD32',  # Verde lima
                '#228B22',  # Verde
                '#006400'   # Verde oscuro (gran mejora)
            ]
        }
        
        difference_tile_url = difference_clipped.getMapId(diff_viz)['tile_fetcher'].url_format
        
        # Visualización del cambio porcentual
        percent_viz = {
            'min': -50,
            'max': 50,
            'palette': diff_viz['palette']
        }
        
        percent_tile_url = percent_clipped.getMapId(percent_viz)['tile_fetcher'].url_format
        
        # Clasificar áreas por magnitud de cambio
        # Crear máscaras para diferentes categorías
        high_degradation = difference.lt(-0.15)  # < -0.15
        moderate_degradation = difference.lt(-0.05).And(difference.gte(-0.15))  # -0.15 a -0.05
        stable = difference.gte(-0.05).And(difference.lte(0.05))  # -0.05 a 0.05
        moderate_improvement = difference.gt(0.05).And(difference.lte(0.15))  # 0.05 a 0.15
        high_improvement = difference.gt(0.15)  # > 0.15
        
        # Calcular áreas de cada categoría
        pixel_area = ee.Image.pixelArea()
        
        def calculate_area(mask):
            area = pixel_area.updateMask(mask).reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=geometry_ee,
                scale=10,
                maxPixels=1e9,
                bestEffort=True
            )
            return area.get('area').getInfo()
        
        area_high_degradation = calculate_area(high_degradation)
        area_moderate_degradation = calculate_area(moderate_degradation)
        area_stable = calculate_area(stable)
        area_moderate_improvement = calculate_area(moderate_improvement)
        area_high_improvement = calculate_area(high_improvement)
        
        # Convertir a hectáreas
        change_areas = {
            'high_degradation': {
                'area_ha': round(area_high_degradation / 10000, 2) if area_high_degradation else 0,
                'percentage': round((area_high_degradation / area_m2) * 100, 2) if area_high_degradation else 0
            },
            'moderate_degradation': {
                'area_ha': round(area_moderate_degradation / 10000, 2) if area_moderate_degradation else 0,
                'percentage': round((area_moderate_degradation / area_m2) * 100, 2) if area_moderate_degradation else 0
            },
            'stable': {
                'area_ha': round(area_stable / 10000, 2) if area_stable else 0,
                'percentage': round((area_stable / area_m2) * 100, 2) if area_stable else 0
            },
            'moderate_improvement': {
                'area_ha': round(area_moderate_improvement / 10000, 2) if area_moderate_improvement else 0,
                'percentage': round((area_moderate_improvement / area_m2) * 100, 2) if area_moderate_improvement else 0
            },
            'high_improvement': {
                'area_ha': round(area_high_improvement / 10000, 2) if area_high_improvement else 0,
                'percentage': round((area_high_improvement / area_m2) * 100, 2) if area_high_improvement else 0
            }
        }
        
        # Extraer valores del píxel con mayor degradación y mejora
        min_value = diff_stats.get('difference_min')
        max_value = diff_stats.get('difference_max')
        
        return jsonify({
            'status': 'success',
            'index': index_name,
            'baseline': {
                'period': f"{baseline_start} a {baseline_end}",
                'images': baseline_count
            },
            'comparison': {
                'period': f"{comparison_start} a {comparison_end}",
                'images': comparison_count
            },
            'difference_map': {
                'tile_url': difference_tile_url,
                'visualization': diff_viz,
                'description': 'Diferencia absoluta (Comparación - Baseline)'
            },
            'percent_change_map': {
                'tile_url': percent_tile_url,
                'visualization': percent_viz,
                'description': 'Cambio porcentual (%)'
            },
            'statistics': {
                'mean_change': round(diff_stats.get('difference_mean'), 4) if diff_stats.get('difference_mean') else None,
                'min_change': round(min_value, 4) if min_value else None,
                'max_change': round(max_value, 4) if max_value else None,
                'std_change': round(diff_stats.get('difference_stdDev'), 4) if diff_stats.get('difference_stdDev') else None,
                'p10': round(diff_stats.get('difference_p10'), 4) if diff_stats.get('difference_p10') else None,
                'p25': round(diff_stats.get('difference_p25'), 4) if diff_stats.get('difference_p25') else None,
                'p50': round(diff_stats.get('difference_p50'), 4) if diff_stats.get('difference_p50') else None,
                'p75': round(diff_stats.get('difference_p75'), 4) if diff_stats.get('difference_p75') else None,
                'p90': round(diff_stats.get('difference_p90'), 4) if diff_stats.get('difference_p90') else None,
            },
            'change_areas': change_areas,
            'area_km2': area_km2,
            'interpretation': {
                'high_degradation': '< -0.15 (Degradación severa)',
                'moderate_degradation': '-0.15 a -0.05 (Degradación moderada)',
                'stable': '-0.05 a 0.05 (Sin cambio significativo)',
                'moderate_improvement': '0.05 a 0.15 (Mejora moderada)',
                'high_improvement': '> 0.15 (Mejora significativa)'
            }
        })
        
    except Exception as e:
        print(f"❌ Error en generate_change_map: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': f'Error al generar mapa de cambios: {str(e)}'
        }), 500


# -------------------------------------------------------
# ENDPOINT 12: COMPOSITOR MULTI-ESPECTRAL
# -------------------------------------------------------

COMPOSICIONES_CONFIG = {
    'RGB': {
        'bands': ['B4', 'B3', 'B2'],
        'min': 0.0,
        'max': 0.3,
        'gamma': 1.3
    },
    'Falso Color IR': {
        'bands': ['B8', 'B4', 'B3'],
        'min': 0.0,
        'max': 0.3,
        'gamma': 1.3
    },
    'Falso Color Agricola': {
        'bands': ['B12', 'B8', 'B4'],
        'min': 0.0,
        'max': 0.3,
        'gamma': 1.3
    },
    'Falso Color SWIR': {
        'bands': ['B8', 'B11', 'B4'],
        'min': 0.0,
        'max': 0.3,
        'gamma': 1.3
    },
    'Deteccion Quemado': {
        'bands': ['B12', 'B8A', 'B4'],
        'min': 0.0,
        'max': 0.3,
        'gamma': 1.3
    }
}

INDICES_VISUALIZATION = {
    'NDVI': {
        'min': -0.2,
        'max': 1.0,
        'palette': [
            "FFFFFF", "CE7E45", "DF923D", "F1B555", "FCD163",
            "99B718", "74A901", "66A000", "529400", "3E8601",
            "207401", "056201", "004C00", "002C00", "001500"
        ]
    },
    'NBR': {
        'min': -0.5,
        'max': 1.0,
        'palette': ['ffffff', '7a8737', 'acbe4d', '0ae042', 'fff70b', 'ffaf38', 'ff641b']
    },
    'CIre': {
        'min': 0,
        'max': 3.0,
        'palette': ['8B4513', 'FFFF00', 'ADFF2F', '32CD32', '228B22', '006400']
    },
    'MSI': {
        'min': 0.0,
        'max': 2.5,
        'palette': ['006400', '228B22', '9ACD32', 'FFFF00', 'FFA500', 'FF0000']
    }
}


# ===== FUNCIÓN PARA LIMPIAR NOMBRES DE ARCHIVO =====
def clean_filename(raw_url, index_or_comp_name):
    """
    Convierte nombres largos de GEE en formato limpio:
    De: 20251021T105111_20251021T105829_T30SXH.NBR.tif
    A:  20251021_T30SXH_NBR.tif
    """
    try:
        # Extraer partes del nombre del archivo de la URL
        filename = raw_url.split('/')[-1].split('?')[0]
        
        # Patrón: captura fecha (YYYYMMDD), tile (T30SXH) y extensión
        match = re.search(r'(\d{8})T\d+_\d{8}T\d+_([A-Z0-9]+)\.', filename)
        
        if match:
            date_part = match.group(1)  # 20251021
            tile_part = match.group(2)  # T30SXH
            
            # Formato limpio: YYYYMMDD_TILE_INDEX.tif
            clean_name = f"{date_part}_{tile_part}_{index_or_comp_name}.tif"
            return clean_name
        else:
            # Fallback: si no coincide el patrón, usar nombre genérico
            return f"download_{index_or_comp_name}.tif"
            
    except Exception as e:
        print(f"⚠️ Error limpiando nombre: {e}")
        return f"download_{index_or_comp_name}.tif"


# ===== FUNCIÓN DE SELECCIÓN DE MEJOR IMAGEN (SIN MÁSCARA) =====
def select_best_image_compositor(collection, target_date, geometry):
    """
    Selecciona la imagen más completa y cercana a la fecha objetivo.
    VERSIÓN SIN MÁSCARA DE NUBES para evitar enmascarar zonas urbanas.
    
    Prioriza:
    1. Cercanía a la fecha objetivo
    2. Menor nubosidad
    3. Cobertura del área (usando datos válidos, no máscaras)
    """
    
    def add_metrics(img):
        # Calcular diferencia de fecha
        diff = ee.Number(img.date().difference(target_date, 'day')).abs()
        
        # Calcular cobertura: % de píxeles con datos válidos (sin usar máscara)
        # Usar B4 (Red) como referencia ya que siempre tiene datos
        valid_pixels = img.select('B4').reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=geometry,
            scale=10,
            maxPixels=1e9,
            bestEffort=True
        ).get('B4')
        
        # Área total en píxeles (10m x 10m = 100m² por píxel)
        total_area = geometry.area()
        expected_pixels = total_area.divide(100)
        
        # Porcentaje de cobertura
        coverage = ee.Number(valid_pixels).divide(expected_pixels).multiply(100)
        
        # Nubosidad del metadato
        cloud = img.get('CLOUDY_PIXEL_PERCENTAGE')
        
        return img.set({
            'date_diff': diff,
            'coverage': coverage,
            'cloud': cloud
        })
    
    # Añadir métricas
    collection_with_metrics = collection.map(add_metrics)
    
    # Filtrar imágenes con cobertura mínima del 70% (más permisivo)
    good_coverage = collection_with_metrics.filter(
        ee.Filter.gte('coverage', 70)
    )
    
    # Si no hay imágenes con 70%, usar 50%
    count = good_coverage.size()
    final_collection = ee.Algorithms.If(
        count.gt(0),
        good_coverage,
        collection_with_metrics.filter(ee.Filter.gte('coverage', 50))
    )
    
    final_collection = ee.ImageCollection(final_collection)
    
    # Ordenar por: 1) fecha cercana, 2) menos nubes
    sorted_collection = final_collection.sort('date_diff').sort('cloud')
    
    # Retornar la mejor imagen
    return sorted_collection.first()


# ===== ENDPOINT COMPOSITOR ACTUALIZADO =====
@app.route("/api/compositor", methods=["POST"])
def compositor_multiespectral():
    """
    Genera múltiples composiciones espectrales e índices
    VERSIÓN CORREGIDA: Composiciones RGB unificadas + nombres limpios
    """
    print("🔵 Endpoint /api/compositor llamado")
    
    try:
        data = request.json
        print(f"📥 Datos recibidos: {data}")
        
        geometry = data.get('geometry')
        date_str = data.get('date')
        max_cloud = data.get('max_cloud', 30)
        composiciones_solicitadas = data.get('composiciones', list(COMPOSICIONES_CONFIG.keys()))
        indices_solicitados = data.get('indices', list(INDICES_VISUALIZATION.keys()))
        
        if not all([geometry, date_str]):
            return jsonify({
                'status': 'error',
                'message': 'Faltan parámetros requeridos: geometry y date'
            }), 400
        
        geometry_ee = build_geometry_aoi(geometry)
        area_m2 = geometry_ee.area().getInfo()
        area_km2 = round(area_m2 / 1e6, 4)
        area_ha = round(area_m2 / 10000, 4)
        
        print(f"📐 Área: {area_km2} km²")
        
        if area_km2 > 200:
            return jsonify({
                'status': 'error',
                'message': f'Área demasiado grande ({area_km2:.2f} km²). Límite: 200 km²'
            }), 400
        
        date_ee = ee.Date(date_str)
        
        # Colección con ventana de ±15 días (SIN MÁSCARA DE NUBES)
        col = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")\
            .filterBounds(geometry_ee)\
            .filterDate(date_ee.advance(-15, 'day'), date_ee.advance(15, 'day'))\
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud))\
            .map(add_all_indices)  # ⭐ ELIMINADO: .map(mask_s2_clouds)
        
        imgs_found = col.size().getInfo()
        print(f"🛰️ Imágenes encontradas: {imgs_found}")
        
        if imgs_found == 0:
            return jsonify({
                'status': 'error',
                'message': f'No se encontraron imágenes válidas en ±15 días con menos de {max_cloud}% de nubes'
            }), 404
        
        # Seleccionar la mejor imagen (SIN MÁSCARA)
        print("🔍 Seleccionando la mejor imagen...")
        best_image = select_best_image_compositor(col, date_ee, geometry_ee)
        
        best_image_info = best_image.getInfo()
        if not best_image_info:
            return jsonify({
                'status': 'error',
                'message': 'No se pudo seleccionar una imagen válida'
            }), 404
        
        clipped = best_image.clip(geometry_ee)
        
        # Información de la imagen
        image_date = best_image.date().format("YYYY-MM-dd").getInfo()
        cloud_percentage = best_image.get('CLOUDY_PIXEL_PERCENTAGE').getInfo()
        coverage = best_image.get('coverage').getInfo()
        date_diff = best_image.get('date_diff').getInfo()
        mgrs_tile = best_image.get('MGRS_TILE').getInfo()
        spacecraft = best_image.get('SPACECRAFT_NAME').getInfo()
        
        print(f"✅ Imagen: {image_date} - Cobertura: {coverage:.1f}% - Nubes: {cloud_percentage:.1f}%")
        
        # ===== GENERAR COMPOSICIONES (SOLUCIÓN DEFINITIVA RGB) =====
        composiciones_resultado = {}
        
        for comp_nombre in composiciones_solicitadas:
            if comp_nombre not in COMPOSICIONES_CONFIG:
                print(f"⚠️ Composición no reconocida: {comp_nombre}")
                continue
            
            try:
                config = COMPOSICIONES_CONFIG[comp_nombre]
                
                # Paso 1: Seleccionar bandas y escalar (0-10000 a 0-1)
                comp_img = clipped.select(config['bands']).divide(10000)
                
                # Paso 2: Aplicar corrección gamma si existe
                gamma = config.get('gamma', 1.0)
                if gamma != 1.0:
                    comp_img = comp_img.pow(1.0 / gamma)
                
                # Paso 3: Normalizar al rango [min, max] y escalar a [0, 1]
                # Aplicar stretch de contraste
                min_val = config['min']
                max_val = config['max']
                comp_stretched = comp_img.subtract(min_val).divide(max_val - min_val).clamp(0, 1)
                
                # Paso 4: Convertir a uint8 (0-255) para GeoTIFF RGB estándar
                comp_uint8 = comp_stretched.multiply(255).byte()
                
                # Paso 5: Renombrar bandas a R, G, B para forzar interpretación RGB
                comp_rgb = comp_uint8.rename(['R', 'G', 'B'])
                
                # ⭐ CRÍTICO: Ahora comp_rgb es una imagen de 3 bandas nombradas R, G, B
                # Al descargar, GeoTIFF la interpreta como RGB en lugar de bandas separadas
                
                # Tile URL para visualización en mapa
                tile_url = comp_rgb.getMapId({
                    'bands': ['R', 'G', 'B'],
                    'min': 0,
                    'max': 255
                })['tile_fetcher'].url_format
                
                # Thumbnail
                thumbnail_url = comp_rgb.getThumbURL({
                    'bands': ['R', 'G', 'B'],
                    'min': 0,
                    'max': 255,
                    'dimensions': 512,
                    'region': geometry_ee.bounds().getInfo()['coordinates'],
                    'format': 'png'
                })
                
                # ⭐ URL de descarga GeoTIFF (3 bandas R,G,B unificadas)
                download_url = comp_rgb.getDownloadURL({
                    'bands': ['R', 'G', 'B'],
                    'scale': 10,
                    'crs': 'EPSG:4326',
                    'fileFormat': 'GeoTIFF',
                    'region': geometry_ee.bounds().getInfo()['coordinates']
                })
                
                # ⭐ NOMBRE LIMPIO
                clean_name = clean_filename(download_url, comp_nombre.replace(' ', '_'))
                
                composiciones_resultado[comp_nombre] = {
                    'download_url': download_url,
                    'tile_url': tile_url,
                    'thumbnail_url': thumbnail_url,
                    'bands': config['bands'],
                    'filename': clean_name,
                    'visualization': {
                        'min': config['min'],
                        'max': config['max'],
                        'gamma': config.get('gamma', 1.0)
                    }
                }
                
                print(f"✅ Composición RGB: {comp_nombre} → {clean_name}")
                print(f"   Bandas originales: {config['bands']} → RGB unificado")
                
            except Exception as e:
                print(f"⚠️ Error en composición {comp_nombre}: {e}")
                traceback.print_exc()
                composiciones_resultado[comp_nombre] = {'error': str(e)}
        
        # ===== GENERAR ÍNDICES =====
        indices_resultado = {}
        
        for indice_nombre in indices_solicitados:
            if indice_nombre not in INDICES_VISUALIZATION:
                print(f"⚠️ Índice no reconocido: {indice_nombre}")
                continue
            
            try:
                vis_params = INDICES_VISUALIZATION[indice_nombre]
                indice_img = clipped.select(indice_nombre)
                
                # Estadísticas
                stats = indice_img.reduceRegion(
                    reducer=ee.Reducer.mean().combine(
                        ee.Reducer.minMax(), '', True
                    ).combine(
                        ee.Reducer.stdDev(), '', True
                    ).combine(
                        ee.Reducer.median(), '', True
                    ),
                    geometry=geometry_ee,
                    scale=10,
                    maxPixels=1e13,
                    bestEffort=True
                ).getInfo()
                
                # Tile URL para mapa
                tile_url = indice_img.getMapId(vis_params)['tile_fetcher'].url_format
                
                # URL de descarga
                download_url = indice_img.getDownloadURL({
                    'scale': 10,
                    'crs': 'EPSG:4326',
                    'fileFormat': 'GeoTIFF',
                    'region': geometry_ee.bounds().getInfo()['coordinates']
                })
                
                # Thumbnail
                thumbnail_url = indice_img.getThumbURL({
                    'min': vis_params['min'],
                    'max': vis_params['max'],
                    'palette': vis_params['palette'],
                    'dimensions': 512,
                    'region': geometry_ee.bounds().getInfo()['coordinates'],
                    'format': 'png'
                })
                
                # ⭐ NOMBRE LIMPIO
                clean_name = clean_filename(download_url, indice_nombre)
                
                indices_resultado[indice_nombre] = {
                    'download_url': download_url,
                    'tile_url': tile_url,
                    'thumbnail_url': thumbnail_url,
                    'filename': clean_name,  # ⭐ NUEVO
                    'statistics': {
                        'mean': round(stats.get(f'{indice_nombre}_mean'), 4) if stats.get(f'{indice_nombre}_mean') else None,
                        'min': round(stats.get(f'{indice_nombre}_min'), 4) if stats.get(f'{indice_nombre}_min') else None,
                        'max': round(stats.get(f'{indice_nombre}_max'), 4) if stats.get(f'{indice_nombre}_max') else None,
                        'std': round(stats.get(f'{indice_nombre}_stdDev'), 4) if stats.get(f'{indice_nombre}_stdDev') else None,
                        'median': round(stats.get(f'{indice_nombre}_median'), 4) if stats.get(f'{indice_nombre}_median') else None
                    },
                    'visualization': vis_params
                }
                
                print(f"✅ Índice: {indice_nombre} → {clean_name}")
                
            except Exception as e:
                print(f"⚠️ Error en índice {indice_nombre}: {e}")
                traceback.print_exc()
                indices_resultado[indice_nombre] = {'error': str(e)}
        
        # Bounds para mapa
        try:
            bounds_obj = geometry_ee.bounds().getInfo()
            coords = bounds_obj["coordinates"][0]
            lats = [c[1] for c in coords]
            lons = [c[0] for c in coords]
            bounds = [[min(lats), min(lons)], [max(lats), max(lons)]]
        except Exception:
            bounds = None
        
        # RESPUESTA
        return jsonify({
            'status': 'success',
            'imagery': {
                'image_used': {
                    'date': image_date,
                    'cloud_percentage': round(cloud_percentage, 2) if cloud_percentage else None,
                    'coverage': round(coverage, 1) if coverage else None,
                    'date_difference_days': int(date_diff) if date_diff else None,
                    'tile': mgrs_tile,
                    'satellite': spacecraft
                },
                'images_available': imgs_found,
                'selection_method': 'best_coverage_and_date'
            },
            'geometry': {
                'area_km2': area_km2,
                'area_ha': area_ha,
                'area_m2': round(area_m2, 2),
                'type': geometry.get("geometry", {}).get("type", "Unknown"),
                'bounds': bounds
            },
            'composiciones': composiciones_resultado,
            'indices': indices_resultado,
            'image_date': image_date,
            'images_found': imgs_found,
            'cloud_percentage': round(cloud_percentage, 2) if cloud_percentage else None,
            'area_km2': area_km2
        })
        
    except Exception as e:
        print(f"❌ Error en compositor: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': f'Error al procesar: {str(e)}'
        }), 500


# -------------------------------------------------------
# DEBUG: Listar todas las rutas registradas
# -------------------------------------------------------
print("\n" + "="*60)
print("🔍 RUTAS REGISTRADAS EN FLASK:")
print("="*60)
for rule in app.url_map.iter_rules():
    print(f"  {rule.methods} {rule.rule}")
print("="*60 + "\n")


# -------------------------------------------------------
# Ejecutar aplicación
# -------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)