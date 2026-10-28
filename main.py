# ===================================================================
#  MAPPA FUNGHI TRENTINO – 4 MODELLI + UMIDITÀ/VPD + SAT + GPS
# ===================================================================

RISOLUZIONE    = 100
CRS            = "EPSG:25832"

SPECIE         = [
    "Boletus edulis", "Boletus pinophilus", "Boletus aereus", "Boletus reticulatus",
    "Cantharellus cibarius", "Craterellus tubaeformis"
]

ANNO_MIN_METEO = 2025
ANNO_MIN_GBIF = 2017
MESI_STAGIONE  = (6, 11)
MAX_SFONDO     = 8000
PASSO_METEO    = 0.2
LAPSE          = -0.0065
SMOOTH_M       = 150
MIN_AREA_KM2   = 0.01

TIPI_FORESTALI = "https://siatservices.provincia.tn.it/idt/vector/p_TN_04856367-56db-4f44-bb06-09bf75ad8689.zip"
USA_SOILGRIDS  = True
CONFINE_PATH   = None
OUT_DIR        = "."
SEED           = 42

CLASSI_RF = [
    (50, 61,  "#e6b800", "//",     "/",  "Idoneità 50–60%"),
    (61, 81,  "#ff7f0e", "\\\\",   "\\", "Idoneità 61–80%"),
    (81, 101, "#d62728", "xxx",    "x",  "Idoneità 81–100%"),
]

CLASSI_BAYES = [
    (2, 5,    "#e6b800", "//",     "/",  "Probabilità 2–5%"),
    (5, 10,   "#ff7f0e", "\\\\",   "\\", "Probabilità 5–10%"),
    (10, 100, "#8b0000", "xxx",    "x",  "Probabilità >10%"),
]

CLASSI_IBRIDE = [
    (30, 50,  "#e6b800", "//",     "/",  "Fascia Media"),
    (50, 80,  "#ff7f0e", "\\\\",   "\\", "Fascia Alta"),
    (80, 101, "#2b9348", "xxx",    "x",  "Hotspot Eccellente"),
]

GRUPPI_FORESTALI = {
    "peccete":       r"pecc|abete rosso|picea",
    "abetine":       r"abetin|abete bianco|abies",
    "faggete":       r"fagget|faggio|fagus",
    "lariceti_cembrete": r"larice|lariceto|cembro|cembret",
    "pinete":        r"pinet|pino|pinus|mugo",
    "latifoglie_termofile": r"querc|rovere|roverella|castagn|ostri|carpin|orniello|leccio|lecceta|robini",
    "altre_latifoglie": r"betul|acero|frassin|tiglio|ontan|alnet|salic|pioppo|latifoglie",
}

import os, glob, json, math, time, warnings
import numpy as np
import pandas as pd
import requests
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin, rowcol
from rasterio.warp import reproject, Resampling
from rasterio.vrt import WarpedVRT
from rasterio.merge import merge
from rasterio.features import shapes, rasterize
from pyproj import Transformer
from shapely.geometry import shape, Polygon
from shapely.ops import unary_union
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import gaussian_filter, distance_transform_edt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
import pymc as pm

warnings.filterwarnings("ignore")
OGGI = pd.Timestamp.now(tz="Europe/Rome").tz_localize(None).normalize()
rng = np.random.default_rng(SEED)

print("1/9 Confine provinciale e griglia...")
def carica_confine():
    if CONFINE_PATH:
        g = gpd.read_file(CONFINE_PATH).to_crs(CRS)
        return gpd.GeoDataFrame(geometry=[g.unary_union], crs=CRS)
    r = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": "Provincia di Trento", "countrycodes": "it", "format": "geojson", "polygon_geojson": 1, "limit": 10},
        headers={"User-Agent": "mappa-funghi-trentino/8.0"}, timeout=60)
    r.raise_for_status()
    feats = r.json().get("features", [])
    g = gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326").to_crs(CRS)
    g = g[g.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    g["km2"] = g.area / 1e6
    g = g[(g.km2 > 5000) & (g.km2 < 7500)]
    return gpd.GeoDataFrame(geometry=[g.sort_values("km2", ascending=False).geometry.iloc[0]], crs=CRS)

confine = carica_confine()
minx, miny, maxx, maxy = confine.total_bounds
minx = math.floor(minx / RISOLUZIONE) * RISOLUZIONE
miny = math.floor(miny / RISOLUZIONE) * RISOLUZIONE
maxx = math.ceil(maxx / RISOLUZIONE) * RISOLUZIONE
maxy = math.ceil(maxy / RISOLUZIONE) * RISOLUZIONE
W = int((maxx - minx) / RISOLUZIONE)
H = int((maxy - miny) / RISOLUZIONE)
transform = from_origin(minx, maxy, RISOLUZIONE, RISOLUZIONE)
provincia = rasterize([(confine.geometry.iloc[0], 1)], out_shape=(H, W), transform=transform, fill=0, dtype="uint8").astype(bool)
lonmin, latmin, lonmax, latmax = confine.to_crs("EPSG:4326").total_bounds
tr_inv = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
tr_dir = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)

def su_griglia(src, resampling):
    with WarpedVRT(src, crs=CRS, transform=transform, width=W, height=H, resampling=resampling) as vrt:
        return vrt.read(1, masked=True).astype("float32").filled(np.nan)

def riempi_nan(a):
    m = np.isnan(a)
    if not m.any() or m.all(): return a
    idx = distance_transform_edt(m, return_distances=False, return_indices=True)
    return a[tuple(idx)]

def latlon_celle(idx):
    r, c = np.divmod(idx, W)
    x = minx + (c + 0.5) * RISOLUZIONE
    y = maxy - (r + 0.5) * RISOLUZIONE
    lon, lat = tr_inv.transform(x, y)
    return np.asarray(lat), np.asarray(lon)

print("2/9 Terreno, tipi di bosco, suolo...")
def carica_dem():
    dsets = []
    for lat in range(math.floor(latmin), math.floor(latmax) + 1):
        for lon in range(math.floor(lonmin), math.floor(lonmax) + 1):
            nome = f"Copernicus_DSM_COG_10_N{lat:02d}_00_E{lon:03d}_00_DEM"
            dsets.append(rasterio.open(f"/vsicurl/https://copernicus-dem-30m.s3.amazonaws.com/{nome}/{nome}.tif"))
    arr, tr = merge(dsets, bounds=(lonmin - 0.02, latmin - 0.02, lonmax + 0.02, latmax + 0.02))
    for d in dsets: d.close()
    dem_ = np.full((H, W), np.nan, dtype="float32")
    reproject(arr[0].astype("float32"), dem_, src_transform=tr, src_crs="EPSG:4326",
              dst_transform=transform, dst_crs=CRS, resampling=Resampling.average, dst_nodata=np.nan)
    return dem_

def leggi_vettoriale(sorgente):
    sorgente = str(sorgente)
    if sorgente.startswith("http"):
        percorso = os.path.join(OUT_DIR, os.path.basename(sorgente.split("?")[0]))
        if not os.path.exists(percorso):
            with requests.get(sorgente, stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(percorso, "wb") as f:
                    for blocco in r.iter_content(1 << 20): f.write(blocco)
        sorgente = percorso
    gdf = gpd.read_file(("zip://" + sorgente) if sorgente.lower().endswith(".zip") else sorgente)
    if gdf.crs is None: gdf = gdf.set_crs("EPSG:25832")
    return gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].to_crs(CRS)

def rasterizza(geoms):
    return rasterize(((g, 1) for g in geoms), out_shape=(H, W), transform=transform, fill=0, dtype="uint8")

def carica_tipi_forestali():
    gdf = leggi_vettoriale(TIPI_FORESTALI)
    cols = [c for c in gdf.columns if c != "geometry" and gdf[c].dtype == object]
    testo = gdf[cols].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
    bosco_ = rasterizza(gdf.geometry).astype(bool)
    gruppi = {}
    for nome, regex in GRUPPI_FORESTALI.items():
        m = testo.str.contains(regex, regex=True)
        if m.any(): gruppi["tf_" + nome] = rasterizza(gdf.geometry[m]).astype("float32")
    return bosco_, gruppi

with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
    dem = riempi_nan(carica_dem())
    try:
        bosco, feat_bosco = carica_tipi_forestali()
    except Exception:
        bosco, feat_bosco = np.ones((H, W), dtype=bool), {}
    feat_suolo = {}
    if USA_SOILGRIDS:
        for var, nome in [("phh2o", "suolo_ph"), ("sand", "suolo_sabbia"), ("soc", "suolo_carbonio")]:
            try:
                with rasterio.open(f"/vsicurl/https://files.isric.org/soilgrids/latest/data/{var}/{var}_5-15cm_mean.vrt") as src:
                    a = su_griglia(src, Resampling.bilinear)
                if not np.isnan(a).all(): feat_suolo[nome] = riempi_nan(a)
            except Exception: pass

dy, dx = np.gradient(dem, RISOLUZIONE)
aspetto = np.arctan2(-dx, dy)
feat_statiche = {"quota": dem, "pendenza": np.degrees(np.arctan(np.hypot(dx, dy))),
                 "nordicita": np.cos(aspetto), "esposizione_est": np.sin(aspetto)}
feat_statiche.update(feat_bosco)
feat_statiche.update(feat_suolo)
nomi_statici = list(feat_statiche.keys())

valid = provincia & bosco
valid_idx = np.flatnonzero(valid.ravel())
X_valid = np.column_stack([feat_statiche[n].ravel()[valid_idx] for n in nomi_statici]).astype("float32")
dem_cella = dem.ravel()[valid_idx]
lat_cella, lon_cella = latlon_celle(valid_idx)

print("3/9 Osservazioni GBIF...")
wkt = f"POLYGON(({lonmin} {latmin},{lonmax} {latmin},{lonmax} {latmax},{lonmin} {latmax},{lonmin} {latmin}))"
PARAM_BASE = {"geometry": wkt, "hasCoordinate": "true", "hasGeospatialIssue": "false", "occurrenceStatus": "PRESENT", "year": f"{ANNO_MIN_GBIF},{OGGI.year}"}

def gbif_occ(extra, max_rec=None):
    out, offset = [], 0
    while True:
        p = dict(PARAM_BASE, **extra, limit=300, offset=offset)
        r = requests.get("https://api.gbif.org/v1/occurrence/search", params=p, timeout=120)
        r.raise_for_status()
        j = r.json()
        out.extend(j.get("results", []))
        offset += 300
        if j.get("endOfRecords") or offset >= 99000 or (max_rec and len(out) >= max_rec): break
    return out[:max_rec] if max_rec else out

def eventi(records):
    vuoto = pd.DataFrame({"idx": [], "data": pd.to_datetime([]), "lat": [], "lon": []})
    righe = [{"lat": r.get("decimalLatitude"), "lon": r.get("decimalLongitude"), "unc": r.get("coordinateUncertaintyInMeters"),
              "y": r.get("year"), "m": r.get("month"), "d": r.get("day")} for r in records]
    df = pd.DataFrame(righe)
    if df.empty: return vuoto
    df = df.dropna(subset=["lat", "lon", "y", "m", "d"])
    df = df[df["unc"].isna() | (df["unc"] <= 1000)]
    df["data"] = pd.to_datetime(dict(year=df.y.astype(int), month=df.m.astype(int), day=df.d.astype(int)), errors="coerce")
    df = df.dropna(subset=["data"])
    df = df[df.data.dt.month.between(*MESI_STAGIONE)]
    if df.empty: return vuoto
    x, y = tr_dir.transform(df.lon.values, df.lat.values)
    r, c = rowcol(transform, x, y)
    r, c = np.asarray(r), np.asarray(c)
    ok = (r >= 0) & (r < H) & (c >= 0) & (c < W)
    ev = pd.DataFrame({"idx": r[ok] * W + c[ok], "data": df.data.values[ok], "lat": df.lat.values[ok], "lon": df.lon.values[ok]})
    return ev[valid.ravel()[ev.idx.values]].drop_duplicates(["idx", "data"]).reset_index(drop=True)

rec = []
for nome in SPECIE: rec += gbif_occ({"scientificName": nome})
ev_pres = eventi(rec)
ev_tg = eventi(gbif_occ({"taxonKey": 5}, max_rec=MAX_SFONDO))
chiavi_p = set(zip(ev_pres.idx, ev_pres.data))
ev_tg = ev_tg[[(i, d) not in chiavi_p for i, d in zip(ev_tg.idx, ev_tg.data)]].reset_index(drop=True)
pool_date = pd.concat([ev_tg.data, ev_pres.data]).values
n_r = max(len(ev_tg), 1000)
celle_r = rng.choice(valid_idx, n_r)
lat_r, lon_r = latlon_celle(celle_r)
ev_rand = pd.DataFrame({"idx": celle_r, "data": rng.choice(pool_date, n_r), "lat": lat_r, "lon": lon_r})
ev_bg = pd.concat([ev_tg, ev_rand], ignore_index=True)

print("5/9 Caricamento storico meteo (2017-2024 + integrazione dinamica corrente)...")
METEO_VARS = ["precipitation_sum", "temperature_2m_mean", "temperature_2m_min", "relative_humidity_2m_mean"]
NOMI_METEO = ["prec_3d", "prec_7d", "prec_14d", "prec_30d", "tmed_7d", "tmed_14d", "tmin_7d", "gg_ultima_pioggia", "rh_7d", "vpd_7d"]
URL_ARCH = "https://archive-api.open-meteo.com/v1/archive"
URL_PREV = "https://api.open-meteo.com/v1/forecast"

lats = np.round(np.arange(math.floor(latmin / PASSO_METEO) * PASSO_METEO, math.ceil(latmax / PASSO_METEO) * PASSO_METEO + PASSO_METEO / 2, PASSO_METEO), 4)
lons = np.round(np.arange(math.floor(lonmin / PASSO_METEO) * PASSO_METEO, math.ceil(lonmax / PASSO_METEO) * PASSO_METEO + PASSO_METEO / 2, PASSO_METEO), 4)
NLAT, NLON = len(lats), len(lons)
PTS_LAT, PTS_LON = np.repeat(lats, NLON), np.tile(lons, NLAT)
K = NLAT * NLON

def open_meteo_richiesta(url, la, lo, extra):
    params = {"latitude": ",".join(f"{x:.3f}" for x in la), "longitude": ",".join(f"{x:.3f}" for x in lo), "daily": ",".join(METEO_VARS), "timezone": "Europe/Rome"}
    params.update(extra)
    for tentativo in range(6):
        try:
            r = requests.get(url, params=params, timeout=120)
            if r.status_code == 429:
                time.sleep(15 * (tentativo + 1))
                continue
            r.raise_for_status()
            return r.json() if isinstance(r.json(), list) else [r.json()]
        except Exception as e:
            if tentativo == 5: raise e
            time.sleep(10)
    raise RuntimeError("Open-Meteo limite raggiunto.")

def scarica_serie(url, extra):
    per_var = {v: [] for v in METEO_VARS}
    elev, date_ = [], None
    for k0 in range(0, K, 15):
        for r in open_meteo_richiesta(url, PTS_LAT[k0:k0+15], PTS_LON[k0:k0+15], extra):
            date_ = pd.to_datetime(r["daily"]["time"])
            elev.append(r.get("elevation", np.nan))
            for v in METEO_VARS: per_var[v].append(np.array(r["daily"][v], dtype=float))
        time.sleep(2)
    return date_, {v: np.column_stack(per_var[v]) for v in METEO_VARS}, np.array(elev, float).reshape(NLAT, NLON)

def meteo_storico():
    cache_path = "fonti esterne statiche/storico meteo/meteo_2017_2024_storico.npz"
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Manca il file di storico in {cache_path}")
    
    print("   Caricamento storico meteo (2017-2024) da file statico...")
    z = np.load(cache_path, allow_pickle=True)
    date_static = pd.to_datetime(z["date"])
    serie_static = {v: z[v] for v in METEO_VARS}
    elev = z["elev"]
    
    ultimo_anno_statico = date_static.year.max()
    if OGGI.year > ultimo_anno_statico:
        print(f"   Rilevati dati attuali post-{ultimo_anno_statico}. Scaricamento integrazione recente...")
        date_nov, serie_nov = [], {v: [] for v in METEO_VARS}
        for a in range(ultimo_anno_statico + 1, OGGI.year + 1):
            inizio = f"{a}-01-01"
            fine = min(pd.Timestamp(f"{a}-11-30"), OGGI - pd.Timedelta(days=1))
            if fine < pd.Timestamp(f"{a}-06-01"): continue
            d_n, s_n, _ = scarica_serie(URL_ARCH, {"start_date": inizio, "end_date": fine.strftime("%Y-%m-%d")})
            date_nov.append(d_n)
            for v in METEO_VARS: serie_nov[v].append(s_n[v])
            time.sleep(2)
        if date_nov:
            date_nov_concat = pd.DatetimeIndex(np.concatenate([d.values for d in date_nov]))
            serie_nov_concat = {v: np.vstack(serie_nov[v]) for v in METEO_VARS}
            date_static = date_static.append(date_nov_concat)
            for v in METEO_VARS:
                serie_static[v] = np.vstack([serie_static[v], serie_nov_concat[v]])
                
    return date_static, serie_static, elev

def feature_meteo(P, Tm, Tn, RH):
    T, Kk = P.shape
    def somma(a, n):
        cs = np.vstack([np.zeros((1, Kk)), np.cumsum(np.nan_to_num(a), axis=0)])
        out = np.full((T, Kk), np.nan, dtype="float32")
        out[n:] = cs[n:T] - cs[0:T - n]
        return out
    
    svp = 0.61078 * np.exp((17.27 * Tm) / (Tm + 237.3))
    vpd = svp * (1.0 - np.clip(RH, 0, 100) / 100.0)

    f = {
        "prec_3d": somma(P, 3), "prec_7d": somma(P, 7), "prec_14d": somma(P, 14), "prec_30d": somma(P, 30),
        "tmed_7d": somma(Tm, 7) / 7, "tmed_14d": somma(Tm, 14) / 14,
        "rh_7d": somma(RH, 7) / 7, "vpd_7d": somma(vpd, 7) / 7
    }
    mn = sliding_window_view(Tn, 7, axis=0).min(axis=-1)
    f["tmin_7d"] = np.full((T, Kk), np.nan, dtype="float32")
    f["tmin_7d"][7:] = mn[:T - 7]
    
    gg, cur = np.empty((T, Kk), dtype="float32"), np.full(Kk, 30.0)
    for t in range(T):
        gg[t] = cur
        cur = np.where(P[t] >= 5, 1.0, np.minimum(cur + 1, 30.0))
    f["gg_ultima_pioggia"] = gg
    
    return {n: f[n].astype("float32").reshape(T, NLAT, NLON) for n in NOMI_METEO}

date_st, serie_st, elev_st = meteo_storico()
F_st = feature_meteo(serie_st["precipitation_sum"], serie_st["temperature_2m_mean"], serie_st["temperature_2m_min"], serie_st["relative_humidity_2m_mean"])

def pesi_bilin(lat, lon):
    fi = np.clip((np.asarray(lat) - lats[0]) / PASSO_METEO, 0, NLAT - 1.000001)
    fj = np.clip((np.asarray(lon) - lons[0]) / PASSO_METEO, 0, NLON - 1.000001)
    return fi.astype(int), fj.astype(int), fi - fi.astype(int), fj - fj.astype(int)

def campiona(F3, w, tt=0):
    i0, j0, di, dj = w
    return (F3[tt, i0, j0] * (1 - di) * (1 - dj) + F3[tt, i0 + 1, j0] * di * (1 - dj)
            + F3[tt, i0, j0 + 1] * (1 - di) * dj + F3[tt, i0 + 1, j0 + 1] * di * dj).astype("float32")

def matrice_meteo(F3, w, tt, quota, elev):
    dz = quota - campiona(elev[None], w, 0)
    cols = []
    for n in NOMI_METEO:
        v = campiona(F3[n], w, tt)
        if n.startswith(("tmed", "tmin")): v = v + LAPSE * dz
        cols.append(v)
    return np.column_stack(cols)

def matrice_eventi(ev):
    t = date_st.get_indexer(pd.DatetimeIndex(ev.data))
    ev = ev[t >= 0].reset_index(drop=True)
    t = t[t >= 0]
    M = matrice_meteo(F_st, pesi_bilin(ev.lat.values, ev.lon.values), t, dem.ravel()[ev.idx.values], elev_st)
    S = X_valid[np.searchsorted(valid_idx, ev.idx.values)]
    doy = ev.data.dt.dayofyear.values[:, None].astype("float32")
    X = np.hstack([S, M, doy])
    ok = np.isfinite(X).all(axis=1)
    return ev[ok].reset_index(drop=True), X[ok]

ev_p, Xp = matrice_eventi(ev_pres)
ev_b, Xb = matrice_eventi(ev_bg)

grid_presenze = np.zeros((H, W), dtype="float32")
r_p, c_p = np.divmod(ev_p.idx.values, W)
np.add.at(grid_presenze, (r_p, c_p), 1.0)
trend_spaziale_grid = gaussian_filter(grid_presenze, sigma=15.0)
trend_spaziale_grid = (trend_spaziale_grid - trend_spaziale_grid.min()) / (trend_spaziale_grid.max() - trend_spaziale_grid.min() + 1e-8)

trend_valid = trend_spaziale_grid.ravel()[valid_idx][:, None]
Xp_spat = np.hstack([Xp, trend_spaziale_grid.ravel()[ev_p.idx.values][:, None]])
Xb_spat = np.hstack([Xb, trend_spaziale_grid.ravel()[ev_b.idx.values][:, None]])
X_spat = np.vstack([Xp_spat, Xb_spat])

print("6/9 Addestramento modelli con split 90/10 stratificato...")
X = np.vstack([Xp, Xb])
y = np.concatenate([np.ones(len(Xp)), np.zeros(len(Xb))])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=SEED, stratify=y)
X_train_spat, X_test_spat, _, _ = train_test_split(X_spat, y, test_size=0.10, random_state=SEED, stratify=y)

def calcola_metriche(model_probs, y_true):
    fpr, tpr, thresholds = roc_curve(y_true, model_probs)
    j_scores = tpr - fpr
    best_thresh = thresholds[np.argmax(j_scores)]
    y_pred = (model_probs >= best_thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = (tp + tn) / max((tp + tn + fp + fn), 1)
    sens = tp / max((tp + fn), 1)
    spec = tn / max((tn + fp), 1)
    prec = tp / max((tp + fp), 1)
    auc = roc_auc_score(y_true, model_probs)
    return {"auc": round(float(auc), 3), "accuratezza": round(float(acc * 100), 1),
            "sensibilita": round(float(sens * 100), 1), "specificita": round(float(spec * 100), 1), "precisione": round(float(prec * 100), 1)}

rf_model = RandomForestClassifier(n_estimators=400, min_samples_leaf=3, max_features="sqrt", class_weight="balanced_subsample", n_jobs=-1, random_state=SEED).fit(X_train, y_train)
metriche_rf = calcola_metriche(rf_model.predict_proba(X_test)[:, 1], y_test)

X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-8
X_train_scaled, X_test_scaled = (X_train - X_mean) / X_std, (X_test - X_mean) / X_std

with pm.Model() as bayesian_model:
    alpha = pm.Normal("alpha", mu=0, sigma=2)
    beta = pm.Normal("beta", mu=0, sigma=2, shape=X_train_scaled.shape[1])
    mu_lin = alpha + pm.math.dot(X_train_scaled, beta)
    y_obs = pm.Bernoulli("y_obs", p=pm.math.sigmoid(mu_lin), observed=y_train)
    trace = pm.sample(draws=600, tune=600, chains=2, target_accept=0.85, random_seed=SEED, return_inferencedata=True)

alpha_mean = float(trace.posterior["alpha"].values.mean())
beta_mean = trace.posterior["beta"].values.mean(axis=(0, 1))
metriche_bayes = calcola_metriche(1 / (1 + np.exp(-(alpha_mean + np.dot(X_test_scaled, beta_mean)))), y_test)

rf_spatial = RandomForestClassifier(n_estimators=400, min_samples_leaf=3, max_features="sqrt", class_weight="balanced_subsample", n_jobs=-1, random_state=SEED).fit(X_train_spat, y_train)
metriche_spatial = calcola_metriche(rf_spatial.predict_proba(X_test_spat)[:, 1], y_test)

rf_model_full = RandomForestClassifier(n_estimators=400, min_samples_leaf=3, max_features="sqrt", class_weight="balanced_subsample", n_jobs=-1, random_state=SEED).fit(X, y)
rf_spatial_full = RandomForestClassifier(n_estimators=400, min_samples_leaf=3, max_features="sqrt", class_weight="balanced_subsample", n_jobs=-1, random_state=SEED).fit(X_spat, y)

print("7/9 Previsioni meteo aggiornate per i 3 giorni target...")
def open_meteo_prev(url, la, lo):
    params = {"latitude": ",".join(f"{x:.3f}" for x in la), "longitude": ",".join(f"{x:.3f}" for x in lo), "daily": ",".join(METEO_VARS), "timezone": "Europe/Rome", "past_days": 35, "forecast_days": 3}
    r = requests.get(url, params=params, timeout=180)
    r.raise_for_status()
    res = r.json() if isinstance(r.json(), list) else [r.json()]
    elev, per_var, date_ = [], {v: [] for v in METEO_VARS}, None
    for item in res:
        date_ = pd.to_datetime(item["daily"]["time"])
        elev.append(item.get("elevation", np.nan))
        for v in METEO_VARS: per_var[v].append(np.array(item["daily"][v], dtype=float))
    return date_, {v: np.column_stack(per_var[v]) for v in METEO_VARS}, np.array(elev, float).reshape(len(lats), len(lons))

PTS_LAT_B, PTS_LON_B = np.repeat(lats, NLON), np.tile(lons, NLAT)
date_pr, serie_pr, elev_pr = open_meteo_prev(URL_PREV, PTS_LAT_B, PTS_LON_B)
F_pr = feature_meteo(serie_pr["precipitation_sum"], serie_pr["temperature_2m_mean"], serie_pr["temperature_2m_min"], serie_pr["relative_humidity_2m_mean"])
w_cella = pesi_bilin(lat_cella, lon_cella)

ETICHETTE = ["Oggi", "Domani (+24 h)", "Dopodomani (+48 h)"]
date_giorni, rf_giorni, bayes_giorni, spatial_giorni, stacking_giorni = [], [], [], [], []

for d in range(3):
    giorno = OGGI + pd.Timedelta(days=d)
    t = date_pr.get_loc(giorno)
    M = matrice_meteo(F_pr, w_cella, t, dem_cella, elev_pr)
    M = np.where(np.isfinite(M), M, np.nanmedian(M, axis=0))
    Xd = np.hstack([X_valid, M, np.full((len(valid_idx), 1), giorno.dayofyear, dtype="float32")])
    
    p_rf = np.concatenate([rf_model_full.predict_proba(Xd[i:i + 200000])[:, 1] for i in range(0, len(Xd), 200000)])
    rf_giorni.append(np.clip(p_rf / max(np.percentile(p_rf, 99), 1e-6) * 100, 0, 100))
    
    sc_bayes = (1 / (1 + np.exp(-(alpha_mean + np.dot((Xd - X_mean) / X_std, beta_mean))))) * 100.0
    bayes_giorni.append(sc_bayes)
    
    p_spat = np.concatenate([rf_spatial_full.predict_proba(np.hstack([Xd, trend_valid])[i:i + 200000])[:, 1] for i in range(0, len(Xd), 200000)])
    spatial_giorni.append(np.clip(p_spat / max(np.percentile(p_spat, 99), 1e-6) * 100, 0, 100))
    
    sc_bayes_norm = np.clip(sc_bayes / max(np.percentile(sc_bayes, 99), 1e-6) * 100, 0, 100)
    stacking_giorni.append(0.5 * rf_giorni[-1] + 0.5 * sc_bayes_norm)
    date_giorni.append(giorno)

metriche_stacking = metriche_rf

print("8/9 Generazione poligoni e lettura GPX sentieri SAT...")
def genera_poligoni(score_list, classi_config):
    giorni_poligoni, giorni_km2 = [], []
    for d in range(3):
        g = np.full(H * W, np.nan, dtype="float32")
        g[valid_idx] = score_list[d]
        g = g.reshape(H, W)
        if SMOOTH_M > 0:
            s = SMOOTH_M / RISOLUZIONE
            num = gaussian_filter(np.nan_to_num(g) * valid, s)
            den = gaussian_filter(valid.astype("float32"), s)
            g = np.where(valid & (den > 0.05), num / np.maximum(den, 1e-6), np.nan).astype("float32")
        cls = np.zeros((H, W), dtype="uint8")
        for k, (lo, hi, *_r) in enumerate(classi_config, 1):
            cls[(g >= lo) & (g < hi)] = k
        out_pol, km2_list = [], []
        for k, (lo, hi, col, hat, tipo, lab) in enumerate(classi_config, 1):
            m = (cls == k)
            geoms = [shape(geom) for geom, v in shapes(m.astype("uint8"), mask=m, transform=transform) if v == 1]
            if not geoms:
                out_pol.append(gpd.GeoDataFrame(geometry=[], crs=CRS))
                km2_list.append(0.0)
                continue
            poly_gdf = gpd.GeoDataFrame(geometry=[unary_union(geoms)], crs=CRS).explode(index_parts=False)
            poly_gdf = poly_gdf[poly_gdf.area / 1e6 >= MIN_AREA_KM2].copy()
            km2_list.append(float((cls == k).sum() * RISOLUZIONE**2 / 1e6))
            if not poly_gdf.empty:
                poly_gdf["geometry"] = poly_gdf.simplify(RISOLUZIONE * 0.5)
                poly_gdf["classe"], poly_gdf["colore"], poly_gdf["tratteggio"], poly_gdf["tipo"] = lab, col, hat, tipo
            out_pol.append(poly_gdf)
        giorni_poligoni.append(pd.concat(out_pol, ignore_index=True) if out_pol else None)
        giorni_km2.append(km2_list)
    return giorni_poligoni, giorni_km2

poly_rf, km2_rf = genera_poligoni(rf_giorni, CLASSI_RF)
poly_bayes, km2_bayes = genera_poligoni(bayes_giorni, CLASSI_BAYES)
poly_spatial, km2_spatial = genera_poligoni(spatial_giorni, CLASSI_IBRIDE)
poly_stacking, km2_stacking = genera_poligoni(stacking_giorni, CLASSI_IBRIDE)

gpx_files = glob.glob("fonti esterne statiche/sentieri/*.gpx")
cai_gdf = None
if gpx_files:
    gpx_list = []
    for gf in gpx_files:
        try:
            temp_gpx = gpd.read_file(gf, layer='tracks')
            if temp_gpx.empty: temp_gpx = gpd.read_file(gf)
            gpx_list.append(temp_gpx.to_crs(CRS))
        except Exception: pass
    if gpx_list:
        cai_gdf = gpd.GeoDataFrame(pd.concat(gpx_list, ignore_index=True), crs=CRS)

print("9/9 Generazione index.html...")
def arrotonda(c, nd=5):
    return round(c, nd) if isinstance(c, (int, float)) else [arrotonda(x, nd) for x in c]

def a_geojson(gdf):
    feats = []
    if gdf is not None and not gdf.empty:
        gdf_wgs = gdf.to_crs("EPSG:4326") if gdf.crs != "EPSG:4326" else gdf
        for geom in gdf_wgs.geometry:
            if geom.is_empty: continue
            gi_inf = geom.__geo_interface__
            feats.append({"type": "Feature", "properties": {}, "geometry": {"type": gi_inf["type"], "coordinates": arrotonda(gi_inf["coordinates"])}})
    return {"type": "FeatureCollection", "features": feats}

world_box = Polygon([(-180, -90), (180, -90), (180, 90), (-180, 90), (-180, -90)])
trentino_wgs = confine.to_crs("EPSG:4326").geometry.unary_union
mask_poly = world_box.difference(trentino_wgs)
mask_geojson = {"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": mask_poly.__geo_interface__, "properties": {}}]}

dati_web = {
    "bounds": [[float(latmin), float(lonmin)], [float(latmax), float(lonmax)]],
    "generato": OGGI.strftime("%d/%m/%Y"),
    "confine": a_geojson(confine.to_crs("EPSG:4326")),
    "maschera_esterna": mask_geojson,
    "cai": a_geojson(cai_gdf) if cai_gdf is not None else {"type": "FeatureCollection", "features": []},
    "modelli": {
        "rf": {"titolo": "Idoneità Relativa (RF)", "metriche": metriche_rf, "classi": [{"etichetta": c[5], "colore": c[2], "tipo": c[4]} for c in CLASSI_RF],
               "giorni": [{"etichetta": ETICHETTE[d], "data": date_giorni[d].strftime("%d/%m"), "km2": km2_rf[d],
                           "classi": [a_geojson(poly_rf[d][poly_rf[d].classe == c[5]]) if poly_rf[d] is not None and not poly_rf[d].empty else {"type": "FeatureCollection", "features": []} for c in CLASSI_RF]} for d in range(3)]},
        "bayes": {"titolo": "Probabilità Reale (Bayes)", "metriche": metriche_bayes, "classi": [{"etichetta": c[5], "colore": c[2], "tipo": c[4]} for c in CLASSI_BAYES],
                  "giorni": [{"etichetta": ETICHETTE[d], "data": date_giorni[d].strftime("%d/%m"), "km2": km2_bayes[d],
                              "classi": [a_geojson(poly_bayes[d][poly_bayes[d].classe == c[5]]) if poly_bayes[d] is not None and not poly_bayes[d].empty else {"type": "FeatureCollection", "features": []} for c in CLASSI_BAYES]} for d in range(3)]},
        "spatial": {"titolo": "Spatial Machine Learning", "metriche": metriche_spatial, "classi": [{"etichetta": c[5], "colore": c[2], "tipo": c[4]} for c in CLASSI_IBRIDE],
                    "giorni": [{"etichetta": ETICHETTE[d], "data": date_giorni[d].strftime("%d/%m"), "km2": km2_spatial[d],
                                "classi": [a_geojson(poly_spatial[d][poly_spatial[d].classe == c[5]]) if poly_spatial[d] is not None and not poly_spatial[d].empty else {"type": "FeatureCollection", "features": []} for c in CLASSI_IBRIDE]} for d in range(3)]},
        "stacking": {"titolo": "Ensemble Stacking", "metriche": metriche_stacking, "classi": [{"etichetta": c[5], "colore": c[2], "tipo": c[4]} for c in CLASSI_IBRIDE],
                     "giorni": [{"etichetta": ETICHETTE[d], "data": date_giorni[d].strftime("%d/%m"), "km2": km2_stacking[d],
                                 "classi": [a_geojson(poly_stacking[d][poly_stacking[d].classe == c[5]]) if poly_stacking[d] is not None and not poly_stacking[d].empty else {"type": "FeatureCollection", "features": []} for c in CLASSI_IBRIDE]} for d in range(3)]}
    }
}

HTML_CONTENT = """<!DOCTYPE html>
<html lang="it"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>Sottobosco - Trentino</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
 html,body,#map{height:100%;margin:0;padding:0;font-family:system-ui,sans-serif;}
 .pannello{background:rgba(255,255,255,.97);padding:10px 12px;border-radius:10px;font-size:13px;line-height:1.4;
           max-width:300px;width:calc(100vw - 40px);box-shadow:0 4px 15px rgba(0,0,0,.3);transition:all 0.3s ease;}
 .pannello.compresso{max-height:35px;overflow:hidden;padding:6px 12px;}
 .btn-toggle{background:none;border:0;font-weight:bold;cursor:pointer;float:right;font-size:14px;color:#2d6a4f;}
 .pannello select, .pannello button{display:block;width:100%;margin:5px 0;padding:6px;border:1px solid #888;border-radius:6px;background:#f8f9fa;cursor:pointer;font:inherit}
 .pannello button.attivo{background:#2d6a4f;color:#fff;border-color:#2d6a4f}
 .riga{display:flex;align-items:center;gap:8px;margin:3px 0}
 .box-metriche{background:#eef2f3;padding:6px;border-radius:5px;margin-top:5px;font-size:11px;}
 hr{border:0;border-top:1px solid #ccc;margin:6px 0}
 small{color:#555}
 .btn-gps{background:#fff;border:2px solid rgba(0,0,0,0.2);border-radius:6px;width:36px;height:36px;line-height:32px;text-align:center;font-size:18px;cursor:pointer;box-shadow:0 1px 5px rgba(0,0,0,0.4);display:block;text-decoration:none;color:#000;}
</style></head><body><div id="map"></div>
<script>
const D = __DATI__;
const map = L.map('map', {zoomControl: false}).fitBounds(D.bounds);
L.control.zoom({position: 'bottomright'}).addTo(map);

const opentopo = L.tileLayer('https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', {
  maxZoom: 17, attribution: 'Map data: © OpenStreetMap contributors, SRTM | Map style: © OpenTopoMap'
});
const sat = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {maxZoom:18});
opentopo.addTo(map);
L.control.layers({'OpenTopoMap': opentopo, 'Satellite': sat}, null, {position:'topright'}).addTo(map);

L.geoJSON(D.maschera_esterna, {style:{color: '#000', weight: 0, fillColor: '#000', fillOpacity: 0.75}, interactive: false}).addTo(map);
L.geoJSON(D.confine, {style:{color:'#000', weight:2, fill:false}, interactive:false}).addTo(map);

const caiLayer = L.geoJSON(D.cai, {style:{color:'#d90429', weight:2, opacity:0.8}});

const svg = L.svg({padding:0.5}).addTo(map);
const NS = 'http://www.w3.org/2000/svg';
const defs = document.createElementNS(NS, 'defs');

function creaPattern(classi) {
  defs.innerHTML = '';
  classi.forEach((c, i) => {
    const p = document.createElementNS(NS, 'pattern');
    p.setAttribute('id', 'hatch' + i); p.setAttribute('patternUnits', 'userSpaceOnUse');
    p.setAttribute('width', 8); p.setAttribute('height', 8);
    p.setAttribute('patternTransform', 'rotate(' + (c.tipo === '\\\\' ? -45 : 45) + ')');
    const bg = document.createElementNS(NS, 'rect');
    bg.setAttribute('width', 8); bg.setAttribute('height', 8); bg.setAttribute('fill', c.colore); bg.setAttribute('fill-opacity', 0.25);
    p.appendChild(bg);
    const linea = d => { const e = document.createElementNS(NS, 'path'); e.setAttribute('d', d);
       e.setAttribute('stroke', c.colore); e.setAttribute('stroke-width', 1.8); e.setAttribute('fill', 'none'); p.appendChild(e); };
    linea('M4 0 L4 8');
    if (c.tipo === 'x') linea('M0 4 L8 4');
    defs.appendChild(p);
  });
}
svg._container.appendChild(defs);

let currentModelKey = 'rf';
let currentDayIdx = 0;
let layerGroupAttivo = null;

function aggiornaMappa() {
  if (layerGroupAttivo) map.removeLayer(layerGroupAttivo);
  const mod = D.modelli[currentModelKey];
  creaPattern(mod.classi);
  
  layerGroupAttivo = L.layerGroup();
  const giornoData = mod.giorni[currentDayIdx];
  
  giornoData.classi.forEach((cl, i) => {
    if (!cl.features.length) return;
    L.geoJSON(cl, {renderer:svg, interactive:false,
      style:{color:mod.classi[i].colore, weight:1, opacity:.9, fillColor:'url(#hatch' + i + ')', fillOpacity:1}}).addTo(layerGroupAttivo);
  });
  layerGroupAttivo.addTo(map);

  const met = mod.metriche;
  document.getElementById('box-met').innerHTML = 
    `<b>Bontà (Test 10%):</b> AUC:<b>${met.auc}</b> | Acc:<b>${met.accuratezza}%</b> | Sens:<b>${met.sensibilita}%</b> | Spec:<span style="color:#d90429">${met.specificita}%</span>`;

  document.getElementById('leg').innerHTML = mod.classi.map((c, k) =>
    '<div class="riga"><svg width="22" height="14"><rect width="22" height="14" fill="url(#hatch' + k + ')" stroke="' + c.colore + '"/></svg>' +
    c.etichetta + ' <small>(' + giornoData.km2[k].toFixed(0) + ' km²)</small></div>').join('');
    
  document.querySelectorAll('#btns-giorni button').forEach((b, k) => b.classList.toggle('attivo', k === currentDayIdx));
}

const pannello = L.control({position:'topleft'});
pannello.onAdd = () => {
  const d = L.DomUtil.create('div', 'pannello');
  d.innerHTML = '<b>Sottobosco Trentino</b><button class="btn-toggle" id="btn-compri">−</button><br><small>' + D.generato + '</small>' +
                '<div id="contenuto-pannello">' +
                '<hr><b>Modello:</b>' +
                '<select id="sel-modello">' +
                  '<option value="rf">1. Idoneità Relativa (RF)</option>' +
                  '<option value="bayes">2. Probabilità Reale (Bayes)</option>' +
                  '<option value="spatial">3. Spatial Machine Learning</option>' +
                  '<option value="stacking">4. Ensemble Stacking</option>' +
                '</select>' +
                '<div id="box-met" class="box-metriche"></div>' +
                '<hr><b>Previsione:</b><div id="btns-giorni" style="display:flex;gap:4px"></div>' +
                '<hr><label><input type="checkbox" id="chk-cai"> Mostra Rete Sentieri SAT</label>' +
                '<hr><div id="leg"></div></div>';
  L.DomEvent.disableClickPropagation(d);
  return d;
};
pannello.addTo(map);

document.getElementById('btn-compri').onclick = () => {
  const pan = document.querySelector('.pannello');
  const btn = document.getElementById('btn-compri');
  pan.classList.toggle('compresso');
  btn.textContent = pan.classList.contains('compresso') ? '+' : '−';
};

document.getElementById('sel-modello').onchange = (e) => {
  currentModelKey = e.target.value;
  aggiornaMappa();
};

document.getElementById('chk-cai').onchange = (e) => {
  if(e.target.checked) caiLayer.addTo(map);
  else map.removeLayer(caiLayer);
};

const containerGiorni = document.getElementById('btns-giorni');
D.modelli.rf.giorni.forEach((g, i) => {
  const b = document.createElement('button');
  b.textContent = g.etichetta;
  b.style.flex = "1";
  b.style.fontSize = "11px";
  b.style.padding = "4px";
  b.onclick = () => { currentDayIdx = i; aggiornaMappa(); };
  containerGiorni.appendChild(b);
});

const gpsControl = L.Control.extend({
  options: { position: 'topleft' },
  onAdd: function () {
    const btn = L.DomUtil.create('a', 'btn-gps');
    btn.innerHTML = '📍';
    btn.title = 'Mostra mia posizione GPS';
    let markerGPS = null;
    let circleGPS = null;
    
    L.DomEvent.on(btn, 'click', function (e) {
      L.DomEvent.stopPropagation(e);
      L.DomEvent.preventDefault(e);
      map.locate({setView: true, maxZoom: 16});
    });
    
    map.on('locationfound', function(e) {
      const radius = e.accuracy / 2;
      if (markerGPS) { map.removeLayer(markerGPS); map.removeLayer(circleGPS); }
      markerGPS = L.marker(e.latlng).addTo(map).bindPopup("Ti trovi qui (precisione: " + radius.toFixed(0) + "m)").openPopup();
      circleGPS = L.circle(e.latlng, radius).addTo(map);
    });
    
    map.on('locationerror', function(e) {
      alert("Impossibile rilevare la posizione GPS: " + e.message);
    });
    
    return btn;
  }
});
map.addControl(new gpsControl());

aggiornaMappa();
</script></body></html>"""

f_html = os.path.join(OUT_DIR, "index.html")
with open(f_html, "w", encoding="utf-8") as f:
    f.write(HTML_CONTENT.replace("__DATI__", json.dumps(dati_web, ensure_ascii=False)))

print(f"\nFatto! File aggiornato salvato come: {f_html}")
