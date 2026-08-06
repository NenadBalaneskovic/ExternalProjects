import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------
# Synthetic vocabularies (based on DrivenData docs)
# ---------------------------------------------
BASINS = ["Lake Victoria", "Lake Tanganyika", "Lake Nyasa", "Ruvuma", "Pangani", "Wami/Ruvu"]
REGIONS = ["Arusha", "Dar es Salaam", "Dodoma", "Kilimanjaro", "Mwanza", "Morogoro"]
LGA = ["Hai", "Moshi", "Arusha DC", "Ilala", "Temeke", "Kinondoni"]
WARD = ["Machame Uroki", "Kilimanjaro Central", "Moshono", "Kijitonyama", "Sinza"]
EXTRACTION = ["gravity", "submersible", "handpump", "rope pump", "motorpump"]
MANAGEMENT = ["water board", "vwc", "private operator", "user-group"]
PAYMENT = ["never pay", "pay per bucket", "monthly", "other"]
WATER_QUALITY = ["soft", "salty", "milky", "fluoride", "unknown"]
QUANTITY = ["enough", "insufficient", "dry", "seasonal"]
SOURCE = ["spring", "rainwater harvesting", "shallow well", "borehole"]
SOURCE_CLASS = ["groundwater", "surface", "unknown"]
WPT_TYPE = ["communal standpipe", "hand pump", "improved spring", "dam"]

LABELS = ["functional", "functional needs repair", "non functional"]


# ---------------------------------------------
# Helper functions
# ---------------------------------------------
def random_date():
    year = np.random.randint(2011, 2014)
    month = np.random.randint(1, 13)
    day = np.random.randint(1, 28)
    return f"{year}-{month:02d}-{day:02d}"


def generate_values(n_rows):
    df = pd.DataFrame({
        "id": np.arange(1, n_rows + 1),
        "amount_tsh": np.random.exponential(scale=300, size=n_rows).round(1),
        "date_recorded": [random_date() for _ in range(n_rows)],
        "funder": np.random.choice(["Government", "World Bank", "Germany", "Private", "unknown"], n_rows),
        "gps_height": np.random.randint(0, 2000, n_rows),
        "installer": np.random.choice(["DWE", "CES", "WE", "unknown"], n_rows),
        "longitude": np.random.uniform(29, 40, n_rows),
        "latitude": np.random.uniform(-12, 0, n_rows),
        "wpt_name": np.random.choice(["Kwa Hassan", "Kwa Mzee", "Kwa Mama", "unknown"], n_rows),
        "num_private": np.random.randint(0, 10, n_rows),
        "basin": np.random.choice(BASINS, n_rows),
        "subvillage": np.random.choice(["A", "B", "C", "D"], n_rows),
        "region": np.random.choice(REGIONS, n_rows),
        "region_code": np.random.randint(1, 30, n_rows),
        "district_code": np.random.randint(1, 10, n_rows),
        "lga": np.random.choice(LGA, n_rows),
        "ward": np.random.choice(WARD, n_rows),
        "population": np.random.randint(0, 500, n_rows),
        "public_meeting": np.random.choice([True, False], n_rows),
        "recorded_by": np.random.choice(["GeoData Consultants Ltd"], n_rows),
        "scheme_management": np.random.choice(MANAGEMENT, n_rows),
        "scheme_name": np.random.choice(["Scheme A", "Scheme B", "unknown"], n_rows),
        "permit": np.random.choice([True, False], n_rows),
        "construction_year": np.random.choice([0, 1980, 1990, 2000, 2010], n_rows),
        "extraction_type": np.random.choice(EXTRACTION, n_rows),
        "extraction_type_group": np.random.choice(EXTRACTION, n_rows),
        "extraction_type_class": np.random.choice(EXTRACTION, n_rows),
        "management": np.random.choice(MANAGEMENT, n_rows),
        "management_group": np.random.choice(["user-group", "commercial", "unknown"], n_rows),
        "payment": np.random.choice(PAYMENT, n_rows),
        "payment_type": np.random.choice(PAYMENT, n_rows),
        "water_quality": np.random.choice(WATER_QUALITY, n_rows),
        "quality_group": np.random.choice(["good", "bad", "unknown"], n_rows),
        "quantity": np.random.choice(QUANTITY, n_rows),
        "quantity_group": np.random.choice(QUANTITY, n_rows),
        "source": np.random.choice(SOURCE, n_rows),
        "source_type": np.random.choice(SOURCE, n_rows),
        "source_class": np.random.choice(SOURCE_CLASS, n_rows),
        "waterpoint_type": np.random.choice(WPT_TYPE, n_rows),
        "waterpoint_type_group": np.random.choice(WPT_TYPE, n_rows),
    })

    # Introduce missing values and noise
    for col in ["gps_height", "population", "construction_year"]:
        mask = np.random.rand(n_rows) < 0.1
        df.loc[mask, col] = 0

    return df


def generate_labels(values_df):
    n = len(values_df)
    # Simple synthetic rule-based labeling
    labels = []
    for _, row in values_df.iterrows():
        if row["quantity"] == "dry" or row["water_quality"] == "milky":
            labels.append("non functional")
        elif row["permit"] is False or row["scheme_name"] == "unknown":
            labels.append("functional needs repair")
        else:
            labels.append("functional")
    return pd.DataFrame({"id": values_df["id"], "status_group": labels})


# ---------------------------------------------
# Main generator
# ---------------------------------------------
def generate_fake_dataset(n_train=8000, n_test=3000, out_dir="data"):
    out = Path(out_dir)
    out.mkdir(exist_ok=True)

    train_values = generate_values(n_train)
    train_labels = generate_labels(train_values)
    test_values = generate_values(n_test)

    train_values.to_csv(out / "TrainingSetValues.csv", index=False)
    train_labels.to_csv(out / "TrainingSetLabels.csv", index=False)
    test_values.to_csv(out / "TestSetValues.csv", index=False)

    print(f"Generated synthetic dataset in {out.resolve()}")


if __name__ == "__main__":
    generate_fake_dataset()