import os, glob, numpy as np, pandas as pd
from pathlib import Path
DATA_ROOT = "Data/EEGIS"        
OUT_ROOT  = "data/pre-processed"
BANDS = ["raw", "delta", "theta", "alpha", "beta", "gamma"]
CANON_CH = ["AF3","F7","F3","FC5","T7","P7","O1","O2","P8","T8","FC6","F4","F8","AF4"]
FS = 128
WIN_SEC = 2.0
HOP_SEC = 0.5
WIN = int(WIN_SEC * FS)
HOP = int(HOP_SEC * FS)
MASK_14 = CANON_CH
MASK_8  = ["F7","F3","FC5","T7","P7","F4","F8","AF3"]
MASK_4  = ["F7","F3","T7","AF3"]

def _quick_read_first_line(path):
    try:
        with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
            return f.readline().strip()
    except Exception:
        return ""

def _detect_delim(line):
    return ";" if line.count(";") > line.count(",") else ","

def _looks_like_row_index(series):
    """Detects if first column is just 1..N or 0..N-1."""
    s = pd.to_numeric(series, errors="coerce")
    if s.isna().any():
        return False
    vals = s.to_numpy()
    if len(vals) < 3:
        return False
    diffs = np.diff(vals)
    return np.all(diffs == 1) and vals[0] in (0, 1)
def read_class_stream(data_root, band, class_id):
    class_dir = os.path.join(data_root, band, f"class_{class_id}")
    files = sorted(glob.glob(os.path.join(class_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSVs for {band}/class_{class_id} under {data_root}")

    stream_list = []
    for f in files:
        first = _quick_read_first_line(f)
        if not first:
            print(f"[WARN] empty file: {f}")
            continue
        delim = _detect_delim(first)

        df = pd.read_csv(f, sep=delim, engine="python").dropna(axis=1, how="all")
        if df.shape[1] >= 15 and _looks_like_row_index(df.iloc[:, 0]):
            df = df.iloc[:, 1:]  

        df.columns = [str(c).strip() for c in df.columns]
        canon = {c.lower(): c for c in CANON_CH}
        rename, keep = {}, []
        for c in df.columns:
            cl = c.lower()
            if cl in canon:
                rename[c] = canon[cl]
                keep.append(c)
        if keep:
            df = df[keep].rename(columns=rename)
        for ch in CANON_CH:
            if ch not in df.columns:
                df[ch] = 0.0
        df = df.reindex(columns=CANON_CH, fill_value=0.0)

        stream_list.append(df.to_numpy(dtype=np.float32))

    if not stream_list:
        raise RuntimeError(f"[read_class_stream] No valid CSV content in {class_dir}")
    return np.concatenate(stream_list, axis=0)  # [T_total x 14]
def stack_bands_streams(data_root, class_id):
    per_band = []
    for band in BANDS:
        X = read_class_stream(data_root, band, class_id)
        per_band.append(X.T)  # [14 x T]
    return np.concatenate(per_band, axis=0)  # [(6*14) x T]
def sliding_windows(X_fused, win=WIN, hop=HOP):
    F, T = X_fused.shape
    if T < win:
        return np.empty((0, F, win), dtype=np.float32)
    idx, chunks = 0, []
    while idx + win <= T:
        chunks.append(X_fused[:, idx:idx+win])
        idx += hop
    return np.stack(chunks, axis=0).astype(np.float32)
def apply_channel_mask(X_windows, mask_names):
    name_to_idx = {ch: i for i, ch in enumerate(CANON_CH)}
    ch_idx = [name_to_idx[n] for n in mask_names]
    F_full = len(CANON_CH)
    keep = []
    for b in range(len(BANDS)):
        base = b * F_full
        keep.extend([base + i for i in ch_idx])
    keep = np.array(keep, dtype=np.int64)
    return X_windows[:, keep, :]
def main():
    Path(OUT_ROOT).mkdir(parents=True, exist_ok=True)
    X_all, y_all = [], []
    for cls in range(0, 9):
        fused = stack_bands_streams(DATA_ROOT, cls)
        Xw = sliding_windows(fused, WIN, HOP)
        yw = np.full((Xw.shape[0],), cls, dtype=np.int64)
        X_all.append(Xw)
        y_all.append(yw)
        print(f"Class {cls}: {Xw.shape[0]} windows")
    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    # Save 14-channel version
    out14 = os.path.join(OUT_ROOT, "windows_2.0s_0.5s_14ch.npz")
    np.savez_compressed(out14, X=X_all, y=y_all)
    print("Saved:", out14, X_all.shape, y_all.shape)
    # Save 8-channel version
    X8 = apply_channel_mask(X_all, MASK_8)
    out8 = os.path.join(OUT_ROOT, "windows_2.0s_0.5s_8ch.npz")
    np.savez_compressed(out8, X=X8, y=y_all)
    print("Saved:", out8, X8.shape)
    # Save 4-channel version
    X4 = apply_channel_mask(X_all, MASK_4)
    out4 = os.path.join(OUT_ROOT, "windows_2.0s_0.5s_4ch.npz")
    np.savez_compressed(out4, X=X4, y=y_all)
    print("Saved:", out4, X4.shape)
main()
