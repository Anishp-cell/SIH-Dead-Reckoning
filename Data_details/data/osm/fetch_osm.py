"""
Offline Map Preprocessing Script:
Downloads OpenStreetMap road network for the Coventry S1 area once via Overpass API
and caches it locally as JSON for 100% offline runtime execution.
"""

import json
import logging
import urllib.parse
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Bounding box covering all S1 and test regions with safety margin
# S1 GPS ranges: Lat [52.398, 52.420], Lon [-1.604, -1.505]
BBOX = (52.390, -1.615, 52.430, -1.500)

QUERY = f"""[out:json][timeout:60];
(
  way["highway"~"motorway|trunk|primary|secondary|tertiary|residential|unclassified|service|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link|living_street"]({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]});
  >;
);
out body;
"""

def fetch_and_save_osm(output_path: Path):
    logger.info(f"Querying Overpass API for bbox {BBOX}...")
    url = "https://overpass-api.de/api/interpreter"
    data = urllib.parse.urlencode({"data": QUERY}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"User-Agent": "SIH26168-Research-OfflineMap/1.0"}
    )
    
    with urllib.request.urlopen(req, timeout=60) as resp:
        content = resp.read().decode("utf-8")
        parsed = json.loads(content)
        
    elements = parsed.get("elements", [])
    ways = [e for e in elements if e.get("type") == "way"]
    nodes = [e for e in elements if e.get("type") == "node"]
    logger.info(f"Downloaded {len(elements)} elements: {len(ways)} ways, {len(nodes)} nodes.")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2)
    logger.info(f"Saved offline OSM data to {output_path} ({output_path.stat().st_size / 1024:.1f} KB).")

if __name__ == "__main__":
    out_file = Path("Data_details/data/osm/coventry_s1_osm.json")
    fetch_and_save_osm(out_file)
