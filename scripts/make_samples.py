"""Generate the two demo datasets in samples/.

They are synthetic, seeded (identical on every run) and shaped like two popular
Kaggle datasets so the real files can be dropped in instead:

- retail_sales.csv  ~ "Superstore Sales": orders 2022-2025 with region, state,
  category, sales, quantity, discount and profit. Built-in patterns to find:
  ~12% yearly growth, a Sep/Nov/Dec peak, West as the largest region,
  Technology as the largest category, loss-making Tables, and profit falling
  as discount rises.
- census_income.csv ~ "Adult Census Income" (UCI): age, education, occupation,
  hours per week, sex and an income bracket, with "?" marking missing values
  just like the original.

Usage:  python scripts/make_samples.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "samples"

REGIONS = {
    "West": [("California", ["Los Angeles", "San Francisco", "San Diego"]), ("Washington", ["Seattle", "Spokane"]),
             ("Oregon", ["Portland"]), ("Colorado", ["Denver"]), ("Arizona", ["Phoenix"])],
    "East": [("New York", ["New York City", "Buffalo"]), ("Pennsylvania", ["Philadelphia", "Pittsburgh"]),
             ("Massachusetts", ["Boston"]), ("New Jersey", ["Newark"]), ("Ohio", ["Columbus", "Cleveland"])],
    "Central": [("Texas", ["Houston", "Dallas", "Austin"]), ("Illinois", ["Chicago"]), ("Michigan", ["Detroit"]),
                ("Minnesota", ["Minneapolis"]), ("Wisconsin", ["Milwaukee"])],
    "South": [("Florida", ["Miami", "Jacksonville"]), ("Georgia", ["Atlanta"]), ("North Carolina", ["Charlotte"]),
              ("Tennessee", ["Nashville"]), ("Virginia", ["Richmond"])],
}  # fmt: skip
REGION_WEIGHT = {"West": 0.32, "East": 0.29, "Central": 0.22, "South": 0.17}
REGION_DISCOUNT = {"West": 0.08, "East": 0.10, "Central": 0.20, "South": 0.13}

# sub-category: (category, typical unit price, base margin, popularity)
PRODUCTS = {
    "Chairs": ("Furniture", 190, 0.10, 6), "Tables": ("Furniture", 330, -0.06, 3),
    "Bookcases": ("Furniture", 260, 0.02, 2), "Furnishings": ("Furniture", 45, 0.14, 8),
    "Binders": ("Office Supplies", 18, 0.16, 14), "Paper": ("Office Supplies", 14, 0.42, 13),
    "Storage": ("Office Supplies", 70, 0.09, 8), "Art": ("Office Supplies", 9, 0.25, 7),
    "Appliances": ("Office Supplies", 120, 0.17, 4),
    "Phones": ("Technology", 230, 0.13, 8), "Accessories": ("Technology", 80, 0.22, 7),
    "Machines": ("Technology", 550, 0.03, 2), "Copiers": ("Technology", 900, 0.32, 1),
}  # fmt: skip
SEGMENTS = {"Consumer": 0.52, "Corporate": 0.30, "Home Office": 0.18}
SHIP_MODES = {"Standard Class": (0.60, 5), "Second Class": (0.20, 3), "First Class": (0.15, 2), "Same Day": (0.05, 0)}
MONTH_SEASON = np.array([0.80, 0.70, 1.00, 0.90, 0.95, 0.95, 0.90, 0.95, 1.30, 1.00, 1.40, 1.45])


def make_sales(rng: np.random.Generator, n_orders: int = 3400) -> pd.DataFrame:
    days = pd.date_range("2022-01-01", "2025-12-31", freq="D")
    years = (days.year - 2022).to_numpy()
    weight = MONTH_SEASON[days.month - 1] * (1.12**years) * np.where(days.dayofweek >= 5, 0.6, 1.0)
    order_days = rng.choice(days, size=n_orders, p=weight / weight.sum())
    order_days = pd.DatetimeIndex(np.sort(order_days))

    sub_names = list(PRODUCTS)
    pop = np.array([PRODUCTS[s][3] for s in sub_names], dtype=float)
    rows = []
    for i, day in enumerate(order_days):
        region = rng.choice(list(REGION_WEIGHT), p=list(REGION_WEIGHT.values()))
        state, cities = REGIONS[region][rng.integers(len(REGIONS[region]))]
        city = cities[rng.integers(len(cities))]
        segment = rng.choice(list(SEGMENTS), p=list(SEGMENTS.values()))
        mode = rng.choice(list(SHIP_MODES), p=[v[0] for v in SHIP_MODES.values()])
        ship = day + pd.Timedelta(int(SHIP_MODES[mode][1]) + int(rng.integers(0, 2)), unit="D")
        order_id = f"US-{day.year}-{100000 + i * 7 + int(rng.integers(0, 7))}"
        for _ in range(int(rng.choice([1, 2, 3, 4], p=[0.45, 0.30, 0.15, 0.10]))):
            sub = rng.choice(sub_names, p=pop / pop.sum())
            category, price, margin, _ = PRODUCTS[sub]
            qty = int(rng.choice(range(1, 10), p=[0.2, 0.25, 0.2, 0.12, 0.09, 0.06, 0.04, 0.02, 0.02]))
            discount = 0.0
            if rng.random() < 0.45 + REGION_DISCOUNT[region]:
                discount = float(rng.choice([0.1, 0.2, 0.3, 0.4, 0.5], p=[0.25, 0.45, 0.12, 0.1, 0.08]))
            unit = price * rng.lognormal(0, 0.45)
            sales = round(unit * qty * (1 - discount), 2)
            profit_rate = margin + rng.normal(0, 0.06) - 1.25 * max(discount - 0.15, 0) - 0.15 * discount
            rows.append(
                {
                    "Order ID": order_id,
                    "Order Date": day.strftime("%m/%d/%Y"),
                    "Ship Date": ship.strftime("%m/%d/%Y"),
                    "Ship Mode": mode,
                    "Segment": segment,
                    "Region": region,
                    "State": state,
                    "City": city,
                    "Category": category,
                    "Sub-Category": sub,
                    "Sales": sales,
                    "Quantity": qty,
                    "Discount": discount,
                    "Profit": round(sales * profit_rate, 2),
                }
            )
    return pd.DataFrame(rows)


EDUCATION = [  # (name, education_num, share)
    ("Preschool", 1, 0.002), ("1st-4th", 2, 0.005), ("5th-6th", 3, 0.01), ("7th-8th", 4, 0.02), ("9th", 5, 0.016),
    ("10th", 6, 0.028), ("11th", 7, 0.036), ("12th", 8, 0.013), ("HS-grad", 9, 0.323), ("Some-college", 10, 0.224),
    ("Assoc-voc", 11, 0.042), ("Assoc-acdm", 12, 0.033), ("Bachelors", 13, 0.164), ("Masters", 14, 0.053),
    ("Prof-school", 15, 0.018), ("Doctorate", 16, 0.013),
]  # fmt: skip
OCCUPATIONS = {  # occupation: (share, income boost)
    "Prof-specialty": (0.13, 0.9), "Craft-repair": (0.13, 0.0), "Exec-managerial": (0.13, 1.0),
    "Adm-clerical": (0.12, -0.4), "Sales": (0.11, 0.2), "Other-service": (0.10, -1.4), "Machine-op-inspct": (0.06, -0.5),
    "Transport-moving": (0.05, -0.1), "Handlers-cleaners": (0.04, -1.1), "Farming-fishing": (0.03, -0.8),
    "Tech-support": (0.03, 0.4), "Protective-serv": (0.02, 0.4), "Priv-house-serv": (0.01, -2.0),
    "Armed-Forces": (0.01, 0.0),
}  # fmt: skip
WORKCLASS = {"Private": 0.74, "Self-emp-not-inc": 0.08, "Local-gov": 0.07, "State-gov": 0.04,
             "Self-emp-inc": 0.035, "Federal-gov": 0.03, "Without-pay": 0.005}  # fmt: skip
MARITAL = {"Married-civ-spouse": 0.46, "Never-married": 0.33, "Divorced": 0.14, "Separated": 0.03, "Widowed": 0.04}
REGIONS_OF_ORIGIN = {"United-States": 0.90, "Mexico": 0.02, "Philippines": 0.01, "Germany": 0.01, "Canada": 0.01,
                     "India": 0.01, "Other": 0.04}  # fmt: skip


def make_census(rng: np.random.Generator, n: int = 6000) -> pd.DataFrame:
    def pick(d: dict, size: int) -> np.ndarray:
        keys = list(d)
        p = np.array([v[0] if isinstance(v, tuple) else v for v in d.values()], dtype=float)
        return rng.choice(keys, size=size, p=p / p.sum())

    age = np.clip(rng.gamma(3.0, 7.2, n) + 17, 17, 90).astype(int)
    edu_idx = rng.choice(len(EDUCATION), size=n, p=np.array([e[2] for e in EDUCATION]) / sum(e[2] for e in EDUCATION))
    education = np.array([EDUCATION[i][0] for i in edu_idx])
    edu_num = np.array([EDUCATION[i][1] for i in edu_idx])
    occupation = pick(OCCUPATIONS, n)
    sex = rng.choice(["Male", "Female"], size=n, p=[0.67, 0.33])
    marital = pick(MARITAL, n)
    marital[age < 22] = "Never-married"
    hours = np.clip(np.round(rng.normal(40, 11, n) + (sex == "Male") * 3), 1, 99).astype(int)
    capital_gain = np.where(rng.random(n) < 0.08, np.round(rng.lognormal(8.3, 1.0, n)), 0).astype(int)

    logit = (
        -9.2
        + 0.38 * edu_num
        + 0.045 * np.minimum(age, 60)
        + 0.03 * (hours - 40)
        + np.array([OCCUPATIONS[o][1] for o in occupation])
        + 1.4 * (marital == "Married-civ-spouse")
        + 0.35 * (sex == "Male")
        + 1.8 * (capital_gain > 5000)
    )
    high = rng.random(n) < 1 / (1 + np.exp(-logit))

    df = pd.DataFrame(
        {
            "age": age,
            "workclass": pick(WORKCLASS, n),
            "education": education,
            "education_num": edu_num,
            "marital_status": marital,
            "occupation": occupation,
            "sex": sex,
            "hours_per_week": hours,
            "capital_gain": capital_gain,
            "native_country": pick(REGIONS_OF_ORIGIN, n),
            "income": np.where(high, ">50K", "<=50K"),
        }
    )
    # The original marks unknown values with "?" - keep that quirk so the importer has to handle it.
    for col, rate in (("workclass", 0.055), ("occupation", 0.057), ("native_country", 0.018)):
        df.loc[rng.random(n) < rate, col] = "?"
    return df


DEPARTMENTS = {  # department: (share, base salary, attrition base rate)
    "Engineering": (0.30, 98000, 0.10), "Sales": (0.22, 68000, 0.22), "Operations": (0.18, 56000, 0.15),
    "Marketing": (0.10, 72000, 0.14), "Finance": (0.08, 82000, 0.09), "HR": (0.06, 61000, 0.12),
    "Customer Support": (0.06, 45000, 0.27),
}  # fmt: skip


def make_hr(rng: np.random.Generator, n: int = 1200) -> pd.DataFrame:
    """HR attrition (like Kaggle "IBM HR Attrition"). Salaries written as "$85,000" text,
    dates day-first (25/03/2021) and Attrition as Yes/No - the importer must convert all three."""
    depts = list(DEPARTMENTS)
    dept = rng.choice(depts, size=n, p=[DEPARTMENTS[d][0] for d in depts])
    level = rng.choice(["Junior", "Mid", "Senior", "Lead"], size=n, p=[0.35, 0.35, 0.22, 0.08])
    mult = pd.Series(level).map({"Junior": 0.8, "Mid": 1.0, "Senior": 1.3, "Lead": 1.6}).to_numpy()
    salary = np.round(np.array([DEPARTMENTS[d][1] for d in dept]) * mult * rng.lognormal(0, 0.12, n), -2)
    hired = pd.Timestamp("2015-01-01") + pd.to_timedelta(rng.integers(0, 3650, n), unit="D")
    satisfaction = rng.choice([1, 2, 3, 4], size=n, p=[0.15, 0.2, 0.35, 0.3])
    overtime = rng.random(n) < 0.28
    p_leave = (
        np.array([DEPARTMENTS[d][2] for d in dept])
        + 0.12 * overtime
        + 0.06 * (satisfaction == 1)
        - 0.04 * (level == "Lead")
    )
    left = rng.random(n) < np.clip(p_leave, 0.01, 0.9)
    return pd.DataFrame(
        {
            "Employee ID": [f"E{10000 + i}" for i in range(n)],
            "Department": dept,
            "Job Level": level,
            "Gender": rng.choice(["Female", "Male"], size=n, p=[0.45, 0.55]),
            "Age": np.clip(np.round(rng.normal(36, 9, n)), 21, 64).astype(int),
            "Hire Date": hired.strftime("%d/%m/%Y"),
            "Annual Salary": [f"${s:,.0f}" for s in salary],
            "Job Satisfaction": satisfaction,
            "Overtime": np.where(overtime, "Yes", "No"),
            "Performance Rating": rng.choice([2, 3, 4], size=n, p=[0.15, 0.65, 0.2]),
            "Attrition": np.where(left, "Yes", "No"),
        }
    )


def make_traffic(rng: np.random.Generator) -> pd.DataFrame:
    """Daily website traffic by channel, written with ';' separators and decimal commas
    (European Excel export style)."""
    days = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    channels = {"Organic Search": (2200, 0.031), "Paid Search": (1300, 0.042), "Social": (900, 0.012),
                "Email": (400, 0.065), "Direct": (1100, 0.038), "Referral": (350, 0.027)}  # fmt: skip
    rows = []
    for day in days:
        weekday = 0.72 if day.dayofweek >= 5 else 1.0
        growth = 1 + 0.35 * (day.dayofyear / 365)
        promo = 1.6 if day.month == 11 and day.day >= 24 else 1.0  # Black Friday week
        for ch, (base, conv) in channels.items():
            boost = promo * (1.8 if ch == "Email" and promo > 1 else 1.0)
            sessions = int(rng.poisson(base * weekday * growth * boost))
            orders = int(rng.binomial(sessions, conv * (1.15 if promo > 1 else 1.0)))
            revenue = orders * rng.normal(62, 6)
            rows.append(
                {
                    "Date": day.strftime("%Y-%m-%d"),
                    "Channel": ch,
                    "Device": rng.choice(["Mobile", "Desktop", "Tablet"], p=[0.58, 0.36, 0.06]),
                    "Sessions": sessions,
                    "Bounce Rate": f"{rng.normal(0.45, 0.06):.3f}".replace(".", ","),
                    "Orders": orders,
                    "Revenue": f"{revenue:.2f}".replace(".", ","),
                }
            )
    return pd.DataFrame(rows)


def make_house_prices(rng: np.random.Generator, n: int = 1500) -> pd.DataFrame:
    """House sales (like Kaggle "House Prices"): numeric relationships for scatter plots,
    some missing values and a free-text description column."""
    areas = {"Riverside": 1.25, "Old Town": 1.1, "Hillcrest": 1.4, "Northgate": 0.9, "Lakeview": 1.3, "Eastfield": 0.8}
    area = rng.choice(list(areas), size=n)
    sqft = np.round(rng.normal(1650, 520, n).clip(500, 4500))
    beds = np.clip(np.round(sqft / 600 + rng.normal(0, 0.7, n)), 1, 6).astype(int)
    year_built = rng.integers(1950, 2024, n)
    price = (
        60000 + sqft * 165 * np.array([areas[a] for a in area]) + (year_built - 1950) * 900 + rng.normal(0, 30000, n)
    ).round(-3)
    garage = rng.choice(["None", "1-car", "2-car", "3-car"], size=n, p=[0.15, 0.35, 0.42, 0.08]).astype(object)
    garage[rng.random(n) < 0.04] = None
    lot = np.round(rng.normal(7200, 2500, n).clip(1500, 20000)).astype(float)
    lot[rng.random(n) < 0.07] = np.nan
    styles = ["charming", "spacious", "renovated", "sunny", "quiet", "modern"]
    features = ["open-plan kitchen", "large garden", "updated bathrooms", "hardwood floors", "a finished basement"]
    return pd.DataFrame(
        {
            "Listing ID": [f"L-{50000 + i}" for i in range(n)],
            "Neighborhood": area,
            "Sale Date": (pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 730, n), unit="D")).strftime("%Y-%m-%d"),
            "Sale Price": price.astype(int),
            "Living Area (sqft)": sqft.astype(int),
            "Lot Size (sqft)": lot,
            "Bedrooms": beds,
            "Year Built": year_built,
            "Garage": garage,
            "Description": [
                "A {} {}-bedroom home in {} with {} and {}.".format(rng.choice(styles), b, a, *rng.choice(features, 2, replace=False))
                for a, b in zip(area, beds)
            ],
        }
    )


def main() -> int:
    OUT.mkdir(exist_ok=True)
    files = {
        "retail_sales.csv": (make_sales(np.random.default_rng(42)), ","),
        "census_income.csv": (make_census(np.random.default_rng(7)), ","),
        "hr_attrition.csv": (make_hr(np.random.default_rng(11)), ","),
        "website_traffic.csv": (make_traffic(np.random.default_rng(23)), ";"),
        "house_prices.csv": (make_house_prices(np.random.default_rng(5)), ","),
    }
    for name, (df, sep) in files.items():
        df.to_csv(OUT / name, index=False, sep=sep)
        print(f"{name}: {len(df):,} rows")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
