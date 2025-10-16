# gee_service/logic/ndvi_timeseries.py
import ee
import datetime

def month_range(start_date, end_date):
    cur = datetime.date.fromisoformat(start_date).replace(day=1)
    end = datetime.date.fromisoformat(end_date).replace(day=1)
    months = []
    while cur <= end:
        months.append(cur.isoformat())
        # avanzar un mes
        year = cur.year + (cur.month // 12)
        month = (cur.month % 12) + 1
        cur = cur.replace(year=year, month=month)
    return months

def get_ndvi_timeseries(geometry_geojson, start_date="2023-01-01", end_date=None, scale=20):
    if end_date is None:
        end_date = datetime.date.today().isoformat()

    geom = ee.Geometry(geometry_geojson)

    # Base collection (Sentinel-2 SR)
    s2 = ee.ImageCollection("COPERNICUS/S2_SR") \
            .filterBounds(geom) \
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30)) \
            .map(lambda img: img.addBands(img.normalizedDifference(['B8', 'B4']).rename('NDVI')))

    months = month_range(start_date, end_date)
    results = []

    for m in months:
        # periodo: primer día del mes -> primer día del mes siguiente
        year, month = map(int, m.split('-'))
        if month == 12:
            next_month = f"{year+1}-01-01"
        else:
            next_month = f"{year}-{(month+1):02d}-01"

        # composite mensual (median) y seleccionar NDVI
        composite = s2.filterDate(m, next_month).select('NDVI').median()

        # reducir sobre la geometría
        try:
            mean_dict = composite.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geom,
                scale=scale,
                maxPixels=1e13
            ).getInfo()
            value = None
            if isinstance(mean_dict, dict) and 'NDVI' in mean_dict:
                value = mean_dict['NDVI']
        except Exception as e:
            value = None

        results.append({"date": m, "value": value})

    return results
