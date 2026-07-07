# =============================================================================
# ZERVE BLOCK 02 — TERRAIN & HYDROLOGY ENGINE
# -----------------------------------------------------------------------------
# INPUTS  (inherited) : dem            (from block_01_ingest)
# OUTPUTS (downstream) : terrain, hydro, slope, hillshade
# =============================================================================
# Derives slope / aspect / roughness / curvature / hillshade, then the D8
# hydrology stack: filled DEM, flow direction, flow accumulation, TWI and
# drainage density.
from src import terrain_analysis, hydrology

try:                          # use the DEM inherited from the upstream block
    dem
except NameError:             # fallback so the block also runs standalone
    from src import data_loader, utils
    utils.ensure_dirs()
    dem = data_loader.load_dem().elevation

terrain = terrain_analysis.analyze_terrain(dem)
hydro = hydrology.analyze_hydrology(dem, terrain["slope"])

slope = terrain["slope"]
hillshade = terrain["hillshade"]

print(f"[02] slope mean {slope.mean():.2f}°  |  TWI mean {hydro['twi'].mean():.2f}  "
      f"|  flow-acc max {hydro['flow_accumulation'].max():.0f}")
