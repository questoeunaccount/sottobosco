import os
import requests
import math
import time
import numpy as np
import pandas as pd

ANNO_MIN = 2017
ANNO_MAX = 2024
PASSO_METEO = 0.2

METEO_VARS = [
    "precipitation_sum",
    "temperature_2m_mean",
    "temperature_2m_min",
    "relative_humidity_2m_mean"
]

URL_ARCH = "https://archive-api.open-meteo.com/v1/archive"

OUT_DIR = r"C:\Users\Pc\Desktop\sottobosco\fonti esterne statiche\storico meteo"

latmin, latmax, lonmin, lonmax = 45.80, 46.50, 10.50, 11.90

lats = np.round(
    np.arange(
        math.floor(latmin / PASSO_METEO) * PASSO_METEO,
        math.ceil(latmax / PASSO_METEO) * PASSO_METEO + PASSO_METEO / 2,
        PASSO_METEO
    ),
    4
)

lons = np.round(
    np.arange(
        math.floor(lonmin / PASSO_METEO) * PASSO_METEO,
        math.ceil(lonmax / PASSO_METEO) * PASSO_METEO + PASSO_METEO / 2,
        PASSO_METEO
    ),
    4
)

PTS_LAT = np.repeat(lats, len(lons))
PTS_LON = np.tile(lons, len(lats))
K = len(PTS_LAT)


def scarica_storico_locale():
    date_all = []
    serie = {v: [] for v in METEO_VARS}
    elev = None

    for a in range(ANNO_MIN, ANNO_MAX + 1):
        print(f"Scaricamento anno {a}...")

        per_var = {v: [] for v in METEO_VARS}
        elev_list = []
        d_arr = None

        for k0 in range(0, K, 10):
            params = {
                "latitude": ",".join(
                    f"{x:.3f}" for x in PTS_LAT[k0:k0 + 10]
                ),
                "longitude": ",".join(
                    f"{x:.3f}" for x in PTS_LON[k0:k0 + 10]
                ),
                "daily": ",".join(METEO_VARS),
                "timezone": "Europe/Rome",
                "start_date": f"{a}-05-01",
                "end_date": f"{a}-11-30"
            }

            for tentativo in range(5):
                r = requests.get(
                    URL_ARCH,
                    params=params,
                    timeout=120
                )

                if r.status_code == 429:
                    attesa = 10 * (tentativo + 1)
                    print(
                        f"   [429] Rallentamento API, "
                        f"attesa di {attesa} secondi..."
                    )
                    time.sleep(attesa)
                    continue

                r.raise_for_status()
                break
            else:
                raise RuntimeError(
                    f"Limite superato per l'anno {a}"
                )

            risposta = r.json()
            res = risposta if isinstance(risposta, list) else [risposta]

            for item in res:
                d_arr = pd.to_datetime(item["daily"]["time"])
                elev_list.append(item.get("elevation", np.nan))

                for v in METEO_VARS:
                    per_var[v].append(
                        np.array(item["daily"][v], dtype=float)
                    )

            time.sleep(1.5)

        date_all.append(d_arr)

        for v in METEO_VARS:
            serie[v].append(np.column_stack(per_var[v]))

        if elev is None:
            elev = np.array(elev_list, dtype=float).reshape(
                len(lats), len(lons)
            )

        print(f"   Anno {a} completato.")
        time.sleep(3)

    date_all = pd.DatetimeIndex(
        np.concatenate([d.values for d in date_all])
    )

    serie_fin = {
        v: np.vstack(serie[v])
        for v in METEO_VARS
    }

    os.makedirs(OUT_DIR, exist_ok=True)

    out_path = os.path.join(
        OUT_DIR,
        "meteo_2017_2024_storico.npz"
    )

    np.savez_compressed(
        out_path,
        date=date_all.astype(str).values,
        elev=elev,
        **serie_fin
    )

    print(f"\nSalvataggio completato con successo in: {out_path}")


if __name__ == "__main__":
    scarica_storico_locale()