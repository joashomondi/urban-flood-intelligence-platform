import importlib.util
mods = ["numpy", "pandas", "scipy", "sklearn", "matplotlib", "plotly",
        "folium", "joblib", "xgboost", "streamlit", "streamlit_folium",
        "pyarrow", "tabulate", "geopandas", "rasterio", "shapely"]
for m in mods:
    ok = importlib.util.find_spec(m) is not None
    print(f"{m:18} {'OK' if ok else 'MISSING'}")
