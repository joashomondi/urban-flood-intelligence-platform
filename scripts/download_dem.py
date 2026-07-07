"""
download_dem.py
===============
Download real SRTM 30 m elevation for the Nairobi study area and write
``data/raw/dem.tif``.

Primary data source (direct link, no API key)
---------------------------------------------
OpenTopography Cloud-Optimized GeoTIFF (COG) mosaic — streams only the study
window via GDAL ``/vsicurl/``:

    https://opentopography.s3.sdsc.edu/raster/SRTM_GL1/SRTM_GL1_srtm.vrt

Fallback: NASA SRTM GL1 Skadi tiles on AWS Open Data:

    https://s3.amazonaws.com/elevation-tiles-prod/skadi/S01/S01E036.hgt.gz

Optional: OpenTopography REST API (requires free API key in ``OPENTOPOGRAPHY_API_KEY``):

    https://portal.opentopography.org/API/globaldem?demtype=SRTMGL1&...

Usage
-----
    python scripts/download_dem.py
    python scripts/download_dem.py --output data/raw/dem.tif
    set OPENTOPOGRAPHY_API_KEY=your_key && python scripts/download_dem.py --method api
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import RAW_DIR, STUDY_AREA, ensure_dirs, get_logger

log = get_logger("ufip.download")

# Direct public endpoints (documented in README / data/external/sources.md)
OPENTOPO_VRT = (
    "https://opentopography.s3.sdsc.edu/raster/SRTM_GL1/SRTM_GL1_srtm.vrt"
)
SRTM_TILE_AWS = (
    "https://s3.amazonaws.com/elevation-tiles-prod/skadi/S01/S01E036.hgt.gz"
)
OPENTOPO_API = "https://portal.opentopography.org/API/globaldem"


def _write_geotiff(out_path: Path, data, transform, crs, source_tag: str) -> Path:
    import numpy as np
    import rasterio

    data = data.astype("float32")
    data[data == -32768] = np.nan
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": np.nan,
        "compress": "deflate",
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data, 1)
        dst.update_tags(
            AREA_OR_POINT="Area",
            source=source_tag,
            study_area=STUDY_AREA.name,
        )
    valid = data[np.isfinite(data)]
    log.info("Wrote %s  shape=%s  elev=[%.0f, %.0f] m",
             out_path, data.shape, valid.min(), valid.max())
    return out_path


def download_from_opentopo_vrt(out_path: Path) -> Path:
    """Clip SRTM from the OpenTopography COG VRT (recommended — small download)."""
    import numpy as np
    import rasterio
    from rasterio.windows import from_bounds

    west, south, east, north = STUDY_AREA.bounds
    vsi = f"/vsicurl/{OPENTOPO_VRT}"
    log.info("Streaming SRTM clip from OpenTopography COG VRT …")
    log.info("  %s", OPENTOPO_VRT)
    with rasterio.open(vsi) as src:
        window = from_bounds(west, south, east, north, transform=src.transform)
        data = src.read(1, window=window, boundless=True, fill_value=np.nan)
        transform = src.window_transform(window)
        crs = src.crs
    return _write_geotiff(
        out_path, data, transform, crs,
        "NASA SRTM GL1 30m via OpenTopography COG VRT (opentopography.s3.sdsc.edu)",
    )


def download_from_opentopo_api(out_path: Path, api_key: str) -> Path:
    """Download a clipped GeoTIFF via the OpenTopography global DEM API."""
    west, south, east, north = STUDY_AREA.bounds
    url = (
        f"{OPENTOPO_API}?demtype=SRTMGL1"
        f"&south={south}&north={north}&west={west}&east={east}"
        f"&outputFormat=GTiff&API_Key={api_key}"
    )
    log.info("Requesting clipped GeoTIFF from OpenTopography API …")
    tmp = out_path.with_suffix(".download.tif")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(out_path)
    log.info("Saved API download -> %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)
    return out_path


def download_from_aws_tile(out_path: Path) -> Path:
    """Download Skadi tile S01E036, read HGT and clip to study bounds."""
    import gzip
    import numpy as np
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.mask import mask
    from shapely.geometry import box, mapping

    log.info("Downloading Skadi tile from AWS (may be slow on some networks) …")
    log.info("  %s", SRTM_TILE_AWS)
    with tempfile.TemporaryDirectory(prefix="srtm_") as tmp:
        gz_path = Path(tmp) / "S01E036.hgt.gz"
        hgt_path = Path(tmp) / "S01E036.hgt"
        urllib.request.urlretrieve(SRTM_TILE_AWS, gz_path)
        with gzip.open(gz_path, "rb") as gz_in, open(hgt_path, "wb") as hgt_out:
            hgt_out.write(gz_in.read())

        size = int((hgt_path.stat().st_size / 2) ** 0.5)
        raw = hgt_path.read_bytes()
        elev = np.frombuffer(raw, dtype=">i2").reshape(size, size).astype("float32")
        transform = from_bounds(36, -2, 37, -1, size, size)  # S01E036 footprint
        profile = {
            "driver": "GTiff", "height": size, "width": size, "count": 1,
            "dtype": "float32", "crs": "EPSG:4326", "transform": transform,
            "nodata": -32768,
        }
        geom = [mapping(box(*STUDY_AREA.bounds))]
        with rasterio.io.MemoryFile() as mem:
            with mem.open(**profile) as src:
                src.write(elev, 1)
                clipped, clip_transform = mask(src, geom, crop=True, nodata=np.nan)
        return _write_geotiff(
            out_path, clipped[0], clip_transform, "EPSG:4326",
            "NASA SRTM GL1 30m via AWS elevation-tiles-prod/skadi/S01/S01E036",
        )


def download_dem(output: Path | None = None, method: str = "vrt") -> Path:
    """Download real SRTM for the study area."""
    try:
        import rasterio  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("rasterio is required: pip install rasterio") from exc

    ensure_dirs()
    out = Path(output or RAW_DIR / "dem.tif")
    log.info("Study area: %s  bounds=%s", STUDY_AREA.name, STUDY_AREA.bounds)

    if method == "api":
        key = os.environ.get("OPENTOPOGRAPHY_API_KEY", "")
        if not key:
            raise RuntimeError(
                "Set OPENTOPOGRAPHY_API_KEY (free at https://portal.opentopography.org/)"
            )
        return download_from_opentopo_api(out, key)

    if method == "aws":
        return download_from_aws_tile(out)

    # default: OpenTopography VRT (best balance of speed + no key)
    try:
        return download_from_opentopo_vrt(out)
    except Exception as exc:
        log.warning("OpenTopography VRT failed (%s); trying AWS Skadi tile …", exc)
        return download_from_aws_tile(out)


def main():
    ap = argparse.ArgumentParser(description="Download SRTM DEM for Nairobi.")
    ap.add_argument("--output", type=Path, default=RAW_DIR / "dem.tif")
    ap.add_argument(
        "--method", choices=("vrt", "aws", "api"), default="vrt",
        help="vrt=OpenTopography COG (default), aws=Skadi tile, api=OT REST API",
    )
    args = ap.parse_args()
    path = download_dem(args.output, method=args.method)
    print(f"\nReal DEM ready: {path}")
    print("Re-run the pipeline:  python scripts/run_pipeline.py")


if __name__ == "__main__":
    main()
