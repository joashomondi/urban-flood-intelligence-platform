"""
download_rainfall.py
====================
Download real CHIRPS rainfall for the Nairobi study area and write
``data/raw/rainfall.tif``.

Data source (direct link, no API key)
-------------------------------------
CHIRPS v2.0 — Climate Hazards Group InfraRed Precipitation with Station data
(UC Santa Barbara). Global gridded precipitation, monthly / daily, ~5 km:

    https://data.chc.ucsb.edu/products/CHIRPS-2.0/

We download one global monthly GeoTIFF (a representative Nairobi wet-season
month by default), clip it to ``STUDY_AREA.bounds`` and export a small raster
that the pipeline auto-detects.

Usage
-----
    python scripts/download_rainfall.py
    python scripts/download_rainfall.py --year 2024 --month 4
    python scripts/download_rainfall.py --output data/raw/rainfall.tif
"""
from __future__ import annotations

import argparse
import gzip
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import RAW_DIR, STUDY_AREA, ensure_dirs, get_logger

log = get_logger("ufip.rainfall")

# CHIRPS v2.0 global monthly GeoTIFFs (gzip-compressed).
CHIRPS_MONTHLY = (
    "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/tifs/"
    "chirps-v2.0.{year}.{month:02d}.tif.gz"
)


def download_rainfall(output: Path | None = None,
                      year: int = 2024, month: int = 4) -> Path:
    """Download a CHIRPS monthly raster and clip it to the study bounds.

    Streams and decompresses entirely **in memory** (no large disk temp files),
    so it works even on space-constrained hosts: only the tiny clipped output
    (a few KB) is written to disk.
    """
    try:
        import numpy as np
        import rasterio
        from rasterio.mask import mask
        from shapely.geometry import box, mapping
    except ImportError as exc:
        raise RuntimeError("rasterio + shapely required: pip install rasterio shapely") from exc

    ensure_dirs()
    out = Path(output or RAW_DIR / "rainfall.tif")
    log.info("Study area: %s  bounds=%s", STUDY_AREA.name, STUDY_AREA.bounds)
    log.info("CHIRPS month: %04d-%02d (mm/month accumulation)", year, month)

    url = CHIRPS_MONTHLY.format(year=year, month=month)
    log.info("Streaming (in-memory) %s", url)
    with urllib.request.urlopen(url, timeout=300) as resp:
        gz_bytes = resp.read()
    log.info("  downloaded %.1f MB compressed; decompressing in memory", len(gz_bytes) / 1e6)
    tif_bytes = gzip.decompress(gz_bytes)
    del gz_bytes

    geom = [mapping(box(*STUDY_AREA.bounds))]
    with rasterio.io.MemoryFile(tif_bytes) as mem, mem.open() as src:
        clipped, clip_transform = mask(src, geom, crop=True)
        data = clipped[0].astype("float32")
        nodata = src.nodata if src.nodata is not None else -9999.0
        crs = src.crs
    del tif_bytes
    data[data == nodata] = np.nan
    data[data < 0] = np.nan

    out.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff", "height": data.shape[0], "width": data.shape[1],
        "count": 1, "dtype": "float32", "crs": crs,
        "transform": clip_transform, "nodata": np.nan, "compress": "deflate",
    }
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(data, 1)
        dst.update_tags(
            source="CHIRPS v2.0 monthly (data.chc.ucsb.edu)",
            period=f"{year}-{month:02d}",
            units="mm/month",
            study_area=STUDY_AREA.name,
        )

    valid = data[np.isfinite(data)]
    log.info("Wrote %s  shape=%s  rainfall=[%.1f, %.1f] mm/month",
             out, data.shape,
             float(valid.min()) if valid.size else 0.0,
             float(valid.max()) if valid.size else 0.0)
    return out


def main():
    ap = argparse.ArgumentParser(description="Download CHIRPS rainfall for Nairobi.")
    ap.add_argument("--output", type=Path, default=RAW_DIR / "rainfall.tif")
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--month", type=int, default=4, help="1-12 (default: April long-rains)")
    args = ap.parse_args()
    path = download_rainfall(args.output, year=args.year, month=args.month)
    print(f"\nReal CHIRPS rainfall ready: {path}")
    print("Re-run the pipeline:  python scripts/run_pipeline.py")


if __name__ == "__main__":
    main()
