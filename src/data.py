"""Load, check, clean and split the sessions.

    python -m src.data      (rebuilds data/processed/sessions_clean.csv)
"""
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


def file_md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def load_raw(path=config.RAW_FILE, check_hash=True):
    if check_hash:
        md5 = file_md5(path)
        # If the file changed, every number in the report would be out of date, so stop here.
        if md5 != config.RAW_MD5:
            raise ValueError(f"{path} has MD5 {md5}, expected the frozen copy {config.RAW_MD5}")
    return pd.read_csv(path)


def clean(raw):
    df = raw.copy()
    n_before = len(df)
    # Exact duplicates are dropped before the split, so the same row cannot sit in train and test.
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"dropped {n_before - len(df)} duplicate rows, {len(df)} left")
    # Every other month is a three-letter abbreviation.
    df["Month"] = df["Month"].replace({"June": "Jun"})
    df["Weekend"] = df["Weekend"].astype(int)
    df[config.TARGET] = df[config.TARGET].astype(int)
    return df


def load_clean():
    """Cleaned data. Rebuilt from raw if the processed file is missing."""
    if not config.CLEAN_FILE.exists():
        save_clean()
    df = pd.read_csv(config.CLEAN_FILE)
    return df


def save_clean():
    df = clean(load_raw())
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.CLEAN_FILE, index=False)
    return df


def split(df):
    """Stratified random 80/20 split.

    There are no dates, only months, and each session is a different user, so there is no
    time order or user to group by. Stratifying keeps the 15.6% buy rate the same on both sides.
    train.py also runs a month-based check (train on Feb-Oct, test on Nov-Dec) as a second opinion."""
    train, test = train_test_split(df, test_size=config.TEST_SHARE, stratify=df[config.TARGET],
                                   random_state=config.SEED)
    return train.reset_index(drop=True), test.reset_index(drop=True)


if __name__ == "__main__":
    out = save_clean()
    print(out.shape, f"buy rate {out[config.TARGET].mean():.3f}")
