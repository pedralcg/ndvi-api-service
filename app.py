# gee_service/app.py - Backend completo con múltiples endpoints
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
CORS(app, resources={
    r"/api/*": {"origins": "*"},
    r"/map/gee-tile/*": {"origins": "*"}
})

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

def build_buffered_aoi(geojson_feature):
    """Crea geometría con buffer de 1km"""
    geometry_data = geojson_feature.get("geometry")
    properties = geojson_feature.get("properties", {})
    geom_type = geometry_data.get("type", "").lower()

    try:
        if geom_type == "point" and "radius" in properties:
            radius_m = properties["radius"]
            coords = geometry_data["coordinates"]
            point_ee = ee.Geometry.Point(coords)
            final_buffer = radius_m + 1000
            geom_aoi = point_ee.buffer(final_buffer)
            print(f"⚪ Círculo: Buffer = {final_buffer}m")
        else:
            geom_original = ee.Geometry(geometry_data)
            geom_aoi = geom_original.buffer(1000)
            print(f"🟦 {geom_type}: Buffer de 1000m aplicado.")
    except Exception as e:
        print(f"⚠️ Error al construir geometría: {e}")
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

# -------------------------------------------------------
# ENDPOINT 1: NDVI original (ya existente)
# -------------------------------------------------------
@app.route("/api/ndvi", methods=["POST"])
def calculate_ndvi():
    """Endpoint original de cálculo de NDVI"""
    try:
        data = request.json
        date_str = data.get("date")
        geojson_feature = data.get("geometry")

        if not date_str or not geojson_feature:
            return jsonify({
                "status": "error",
                "message": "Faltan parámetros: 'date' o 'geometry'."
            }), 400

        geometry_ee = build_buffered_aoi(geojson_feature)
        area_km2 = geometry_ee.area().divide(1e6).getInfo()
        date_ee = ee.Date(date_str)
        
        col = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")\
            .filterBounds(geometry_ee)\
            .filterDate(date_ee.advance(-1, 'month'), date_ee.advance(1, 'month'))\
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))\
            .map(mask_s2_clouds)\
            .map(lambda img: add_spectral_index(img, 'NDVI'))
        
        imgs_found = col.size().getInfo()
        
        if imgs_found == 0:
            return jsonify({
                "status": "warning",
                "message": "⚠️ No hay imágenes disponibles.",
                "images_found": 0,
            }), 200

        def add_diff(img):
            diff = ee.Number(img.date().difference(date_ee, "day")).abs()
            return img.set("diff", diff)

        nearest = ee.Image(col.map(add_diff).sort("diff").first())
        
        mean_val = nearest.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry_ee,
            scale=10,
            maxPixels=1e13,
            bestEffort=True
        )
        mean_val_numeric = mean_val.get("NDVI").getInfo()
        
        vis_params = {
            "min": 0.0,
            "max": 1.0,
            "palette": [
                "FFFFFF", "CE7E45", "DF923D", "F1B555", "FCD163",
                "99B718", "74A901", "66A000", "529400", "3E8601",
                "207401", "056201", "004C00", "002C00", "001500",
            ],
        }

        clipped_ndvi = nearest.select("NDVI").clip(geometry_ee)
        tile_url = clipped_ndvi.getMapId(vis_params)["tile_fetcher"].url_format
        
        try:
            bounds_obj = geometry_ee.bounds().getInfo()
            coords = bounds_obj["coordinates"][0]
            lats = [c[1] for c in coords]
            lons = [c[0] for c in coords]
            bounds = [[min(lats), min(lons)], [max(lats), max(lons)]]
        except Exception:
            bounds = None

        image_date = nearest.date().format("YYYY-MM-dd").getInfo()

        return jsonify({
            "status": "success",
            "mean_ndvi": round(mean_val_numeric, 4) if mean_val_numeric else None,
            "tile_url": tile_url,
            "bounds": bounds,
            "images_found": imgs_found,
            "image_date": image_date,
            "area_km2": area_km2,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

# -------------------------------------------------------
# ENDPOINT 2: Series Temporales con Tendencia
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
        
        geometry_ee = build_buffered_aoi(geometry)
        
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
        
        # Verificar si hay imágenes
        imgs_count = col.size().getInfo()
        if imgs_count == 0:
            return jsonify({
                "status": "warning",
                "message": "No hay imágenes disponibles en el periodo",
                "images_found": 0
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
        
        # Calcular deltas
        series_list = series.toList(series.size())
        series_size = series.size().getInfo()
        
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
            'palette': ['red', 'white', 'green']
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
# ENDPOINT 3: Análisis con Umbrales
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
        
        geometry_ee = build_buffered_aoi(geometry)
        
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
                "message": "No hay imágenes disponibles"
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
# ENDPOINT 4: Compuesto Temporal
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
        
        geometry_ee = build_buffered_aoi(geometry)
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
                "message": "No hay imágenes en la ventana temporal"
            }), 200
        
        # Imagen mediana
        median_img = col.median().clip(geometry_ee)
        
        # Visualizaciones por índice
        visualizations = {
            'NDVI': {
                'min': 0.0,
                'max': 0.4,
                'palette': ['FF0000', 'FFA500', 'FFFF00', '00FF00']
            },
            'NBR': {
                'min': -0.5,
                'max': 1.0,
                'palette': ['ffffff', '7a8737', 'acbe4d', '0ae042', 'fff70b', 'ffaf38', 'ff641b']
            },
            'CIre': {
                'min': 0,
                'max': 0.5,
                'palette': ['8B4513', 'FFA500', 'FFFF00', '90EE90', '008000']
            },
            'MSI': {
                'min': 0.3,
                'max': 2.0,
                'palette': ['00008B', '0000FF', '00FFFF', 'FFFF00', 'FFA500', 'FF0000']
            }
        }
        
        # Generar tiles y calcular estadísticas
        tiles = {}
        stats = {}
        
        for idx in indices:
            if idx in visualizations:
                # Tile URL
                tile_url = median_img.select(idx).getMapId(
                    visualizations[idx]
                )['tile_fetcher'].url_format
                tiles[idx] = tile_url
                
                # Estadísticas
                idx_stats = median_img.select(idx).reduceRegion(
                    reducer=ee.Reducer.mean().combine(
                        ee.Reducer.minMax(), '', True
                    ),
                    geometry=geometry_ee,
                    scale=10,
                    maxPixels=1e13,
                    bestEffort=True
                ).getInfo()
                
                stats[idx] = {
                    'mean': idx_stats.get(f'{idx}_mean'),
                    'min': idx_stats.get(f'{idx}_min'),
                    'max': idx_stats.get(f'{idx}_max')
                }
        
        return jsonify({
            "status": "success",
            "tiles": tiles,
            "stats": stats,
            "images_used": imgs_count,
            "date_range": {
                'start': start_date.format('YYYY-MM-dd').getInfo(),
                'end': end_date.format('YYYY-MM-dd').getInfo()
            }
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

# -------------------------------------------------------
# ENDPOINT 5: Comparación Multi-índice
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
        
        geometry_ee = build_buffered_aoi(geometry)
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
                "message": "No hay imágenes disponibles"
            }), 200
        
        # Imagen más cercana
        def add_diff(img):
            diff = ee.Number(img.date().difference(date_ee, "day")).abs()
            return img.set("diff", diff)
        
        nearest = ee.Image(col.map(add_diff).sort("diff").first())
        image_date = nearest.date().format("YYYY-MM-dd").getInfo()
        
        # Visualizaciones
        visualizations = {
            'NDVI': {'min': 0.0, 'max': 1.0, 'palette': ['FFFFFF', 'CE7E45', 'DF923D', 'F1B555', 'FCD163', '99B718', '74A901', '66A000', '529400', '3E8601', '207401', '056201', '004C00', '002C00', '001500']},
            'NBR': {'min': -0.5, 'max': 1.0, 'palette': ['ffffff', '7a8737', 'acbe4d', '0ae042', 'fff70b', 'ffaf38', 'ff641b']},
            'CIre': {'min': 0, 'max': 0.5, 'palette': ['8B4513', 'FFA500', 'FFFF00', '90EE90', '008000']},
            'MSI': {'min': 0.3, 'max': 2.0, 'palette': ['00008B', '0000FF', '00FFFF', 'FFFF00', 'FFA500', 'FF0000']}
        }
        
        results = {}
        for idx in indices:
            # Estadísticas
            mean_val = nearest.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry_ee,
                scale=10,
                maxPixels=1e13,
                bestEffort=True
            )
            
            # Tile URL
            clipped = nearest.select(idx).clip(geometry_ee)
            tile_url = clipped.getMapId(visualizations[idx])['tile_fetcher'].url_format
            
            results[idx] = {
                'mean': round(mean_val.get(idx).getInfo(), 4) if mean_val.get(idx).getInfo() else None,
                'tile_url': tile_url
            }
        
        return jsonify({
            "status": "success",
            "image_date": image_date,
            "images_found": imgs_found,
            "results": results
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

# -------------------------------------------------------
# ENDPOINT 6: Detección de Anomalías
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
        
        geometry_ee = build_buffered_aoi(geometry)
        
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
                "message": "No hay imágenes en el periodo de referencia"
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
                "message": "No hay imágenes en la fecha de test"
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
            anomaly_type = "degradation"  # Degradación
        elif z_score > 1.5:
            anomaly_type = "improvement"  # Mejora
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
            "test_image_date": test_img.date().format("YYYY-MM-dd").getInfo()
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
        # Test simple de GEE
        test = ee.Number(1).getInfo()
        return jsonify({
            "status": "healthy",
            "service": "GeoVisor Backend API",
            "gee_initialized": True,
            "version": "2.0.0"
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
# Ejecutar aplicación
# -------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)