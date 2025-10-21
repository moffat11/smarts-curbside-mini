# gis/export_geo.py (unchanged idea, now uses time_sec if present)
import csv, json, os
IN_CSV = "det_track/outputs/tracks_with_speed.csv"
OUT = "gis/outputs/detections.geojson"
os.makedirs("gis/outputs", exist_ok=True)

LON0, LAT0, SCALE = -113.49, 53.54, 1e-5
features = []
with open(IN_CSV, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        cx = (float(row["xmin"]) + float(row["xmax"])) / 2
        cy = (float(row["ymin"]) + float(row["ymax"])) / 2
        lon = LON0 + cx*SCALE; lat = LAT0 - cy*SCALE
        props = {k: row[k] for k in row if k not in ["xmin","ymin","xmax","ymax"]}
        features.append({"type":"Feature",
                         "geometry":{"type":"Point","coordinates":[lon,lat]},
                         "properties":props})
open(OUT,"w",encoding="utf-8").write(json.dumps({"type":"FeatureCollection","features":features}))
print("Wrote:", OUT)
