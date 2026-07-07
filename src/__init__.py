"""
Urban Flood Intelligence Platform (UFIP)
========================================
A real-time geospatial risk-scoring and stormwater-analysis pipeline.

Modular engines
---------------
    utils                - configuration, paths, logging, helpers
    data_loader          - DEM / rainfall / vector ingestion (real + synthetic)
    terrain_analysis     - slope, aspect, roughness, curvature, hillshade
    hydrology            - D8 flow direction, accumulation, TWI, drainage density
    feature_engineering  - normalised flood indicators + synthetic labels
    modeling             - flood-susceptibility classifiers + evaluation
    risk_scoring         - the signature Flood Risk Score (FRS) engine
    visualization        - matplotlib / folium / plotly operational visuals
"""
from __future__ import annotations

__version__ = "1.0.0"
__all__ = [
    "utils", "data_loader", "terrain_analysis", "hydrology",
    "feature_engineering", "modeling", "risk_scoring", "visualization",
    "pipeline",
]
