import os
import pandas as pd
import numpy as np

# Automatically resolve the root data directory relative to this script
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
EXTENSIONS = [".csv", ".xlsx", ".xls"]


def load_file(path):
    ext = os.path.splitext(path)[1].lower()

    if ext == ".csv":
        try:
            return pd.read_csv(path, low_memory=False, encoding="utf-8")
        except UnicodeDecodeError:
            return pd.read_csv(path, low_memory=False, encoding="latin1")

    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)

    return None


def profile_dataset(path):
    try:
        df = load_file(path)

        if df is None:
            return None

        rows, cols = df.shape

        print("\n" + "=" * 90)
        print(f"DATASET: {path}")
        print("=" * 90)

        print(f"Rows       : {rows:,}")
        print(f"Columns    : {cols:,}")
        print(f"Memory     : {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"Duplicates : {df.duplicated().sum():,}")

        print("\nCOLUMNS")
        print("-" * 90)

        profile = pd.DataFrame({
            "column": df.columns,
            "dtype": df.dtypes.astype(str).values,
            "missing": df.isna().sum().values,
            "missing_%": (df.isna().mean() * 100).round(2).values,
            "unique": df.nunique(dropna=True).values
        })

        print(profile.to_string(index=False))

        # Date-like columns
        date_cols = []

        for col in df.columns:
            name = str(col).lower()

            if any(x in name for x in [
                "date", "time", "year", "month"
            ]):
                date_cols.append(col)

        if date_cols:
            print("\nPOSSIBLE DATE COLUMNS")
            print("-" * 90)

            for col in date_cols:
                values = pd.to_datetime(
                    df[col],
                    errors="coerce"
                )

                valid = values.dropna()

                if len(valid):
                    print(
                        f"{col}: "
                        f"{valid.min().date()} -> {valid.max().date()}"
                    )

        # Geographic columns
        geo_cols = []

        for col in df.columns:
            name = str(col).lower()

            if any(x in name for x in [
                "country",
                "state",
                "city",
                "latitude",
                "longitude",
                "lat",
                "lon",
                "location",
                "region"
            ]):
                geo_cols.append(col)

        if geo_cols:
            print("\nPOSSIBLE GEOGRAPHIC COLUMNS")
            print("-" * 90)

            for col in geo_cols:
                print(
                    f"{col}: "
                    f"{df[col].nunique(dropna=True):,} unique values"
                )

        # Numeric summary
        numeric = df.select_dtypes(include=np.number)

        if not numeric.empty:
            print("\nNUMERIC SUMMARY")
            print("-" * 90)

            summary = numeric.describe().T[
                ["count", "mean", "std", "min", "max"]
            ].round(3)

            print(summary.to_string())

        return df

    except Exception as e:
        print(f"\nERROR reading {path}")
        print(e)
        return None


def find_datasets():
    files = []

    for root, _, filenames in os.walk(DATA_DIR):
        for filename in filenames:
            if any(filename.lower().endswith(ext) for ext in EXTENSIONS):
                files.append(os.path.join(root, filename))

    return sorted(files)


def main():
    print("\nSUPPLY CHAIN DATASET AUDIT")
    print("=" * 90)

    files = find_datasets()

    if not files:
        print("No CSV/Excel datasets found inside data/")
        return

    print(f"\nDatasets found: {len(files)}")

    for file in files:
        profile_dataset(file)

    print("\n" + "=" * 90)
    print("AUDIT COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()
