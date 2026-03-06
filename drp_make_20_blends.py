import os
import pandas as pd
import numpy as np

OLD_PATH = r"submissions_batch\archive\submission_V4_4_DRP_tuned_ET_fixorder.csv"
NEW_PATH = r"submissions_batch\archive\submission_V5_2_OOFTE_fixkeys_as_is__Aplus_grid010__20260305_182709.csv"
OUT_DIR  = r"submissions_batch\archive\drp_20"

os.makedirs(OUT_DIR, exist_ok=True)

old = pd.read_csv(OLD_PATH)
new = pd.read_csv(NEW_PATH)

assert list(old.columns) == list(new.columns), "Columnas no coinciden OLD vs NEW"
assert old.shape == new.shape, "Shape no coincide OLD vs NEW"

# detectar DRP
drp_cols = [c for c in old.columns if ("Dissolved" in c) or (c.strip().upper() == "DRP")]
if len(drp_cols) != 1:
    raise ValueError(f"No pude identificar DRP único. Candidatas: {drp_cols}")
DRP = drp_cols[0]

old_drp = old[DRP].astype(float)
new_drp = new[DRP].astype(float)

# clipping robusto basado en OLD (para no meter outliers raros del NEW)
lo, hi = old_drp.quantile(0.002), old_drp.quantile(0.998)

def clip_like_old(x):
    return x.clip(lo, hi)

def rank01(x):
    r = x.rank(method="average")
    return (r - r.min()) / (r.max() - r.min() + 1e-12)

old_rank = rank01(old_drp)
new_rank = rank01(new_drp)

# 20 configs conservadoras:
# - 12 "raw blends" muy pequeños (1–12%)
# - 8 "rank blends" pequeños (1–12%) que a veces generalizan mejor
raw_weights  = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10,0.11,0.12]
rank_weights = [0.01,0.02,0.03,0.04,0.05,0.07,0.10,0.12]

created = []

# RAW
for w in raw_weights:
    out = old.copy()
    out[DRP] = clip_like_old((1-w)*old_drp + w*new_drp)
    fname = f"submission_DRPblend_raw_w{int(w*100):02d}.csv"
    path = os.path.join(OUT_DIR, fname)
    out.to_csv(path, index=False)
    created.append(fname)

# RANK (mezclamos ranks y luego mapeamos a escala OLD por cuantiles)
old_vals = old_drp.to_numpy()
old_vals_sorted = np.sort(old_vals)

def quantile_map01_to_old(q01: np.ndarray) -> np.ndarray:
    q01 = np.clip(q01, 0.0, 1.0)
    # índice en el array ordenado
    idx = (q01 * (len(old_vals_sorted)-1)).round().astype(int)
    return old_vals_sorted[idx]

for w in rank_weights:
    out = old.copy()
    blended_rank = (1-w)*old_rank + w*new_rank
    out[DRP] = clip_like_old(pd.Series(quantile_map01_to_old(blended_rank.to_numpy())))
    fname = f"submission_DRPblend_rank_w{int(w*100):02d}.csv"
    path = os.path.join(OUT_DIR, fname)
    out.to_csv(path, index=False)
    created.append(fname)

print("✅ Listo. Carpeta:", OUT_DIR)
print("DRP col:", DRP)
print("Clip OLD-based:", float(lo), float(hi))
print("Archivos (20):")
for f in created:
    print(" -", f)