import os
import sys
import requests
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO

SAVE_DIR = Path("data/hydrology")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Python-requests"
}

DATASETS = {
    # HydroRIVERS África (shapefile) — este link sí es directo a data.hydrosheds.org
    "hydrorivers_af_shp": "https://data.hydrosheds.org/file/HydroRIVERS/HydroRIVERS_v10_af_shp.zip",

    # HydroATLAS (BasinATLAS shapefile global) — está en Figshare (muy grande ~4GB)
    # URL directa (ndownloader). Si Figshare bloquea, te doy abajo el plan B por navegador.
    "basinatlas_global_shp": "https://figshare.com/ndownloader/files/20087237",
}

def download(url: str, out_path: Path) -> None:
    import time

    print(f"→ Downloading: {url}")
    max_tries = 12  # ~2 minutos si espera 10s
    wait_seconds = 10

    for attempt in range(1, max_tries + 1):
        with requests.get(url, headers=HEADERS, stream=True, allow_redirects=True, timeout=120) as r:
            if r.status_code == 202:
                print(f"   Figshare respondió 202 (preparando archivo). Reintento {attempt}/{max_tries} en {wait_seconds}s...")
                time.sleep(wait_seconds)
                continue

            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code} for {url}")

            total = int(r.headers.get("Content-Length", "0"))
            chunk = 1024 * 1024  # 1MB
            downloaded = 0

            with open(out_path, "wb") as f:
                for part in r.iter_content(chunk_size=chunk):
                    if part:
                        f.write(part)
                        downloaded += len(part)
                        if total > 0:
                            pct = 100.0 * downloaded / total
                            sys.stdout.write(f"\r   {out_path.name}: {pct:5.1f}%")
                            sys.stdout.flush()

            if total > 0:
                sys.stdout.write("\n")
            return

    raise RuntimeError(f"Figshare siguió respondiendo 202 después de {max_tries} intentos: {url}")

def extract_zip(zip_path: Path, out_dir: Path) -> None:
    print(f"→ Extracting: {zip_path.name}")
    with ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)

def main():
    for name, url in DATASETS.items():
        zip_path = SAVE_DIR / f"{name}.zip"
        if zip_path.exists():
            print(f"✓ Exists, skip download: {zip_path}")
        else:
            download(url, zip_path)

        # intenta extraer
        try:
            extract_zip(zip_path, SAVE_DIR)
        except Exception as e:
            print(f"⚠️  Could not extract {zip_path.name}: {e}")
            print("   (Si es un archivo gigante o bloqueado, usa Plan B de abajo.)")

    print("\n✅ Done. Check files in:", SAVE_DIR)

if __name__ == "__main__":
    main()