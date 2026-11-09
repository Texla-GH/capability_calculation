"""Texla process capability analysis - console program.

Run each menu stage in order. All calculations use individual measurements.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import anderson, norm

D2, A2, D3, D4 = 2.326, 0.577, 0.0, 2.114
OUTPUT = Path("outputs")


def load_data(path):
    """Import M3 data without losing leading zeros in identifiers."""
    data = pd.read_excel(path, sheet_name="Export", dtype={
        "item_number": str, "test": str,
        "manufacturing_order_number": str, "quality_request": str,
    })
    columns = ["item_number", "test", "manufacturing_order_number",
               "measurement_type", "test_result", "minimum_expected",
               "maximum_expected", "registration_date"]
    missing = [column for column in columns if column not in data]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    data["registration_date"] = pd.to_datetime(data["registration_date"], errors="coerce")
    return data


def select_data(data, item, test):
    """Select measurements and recorded averages for an item/test."""
    result = data[(data["item_number"] == item) & (data["test"] == test)].copy()
    if result.empty:
        raise ValueError("No data for this item/test.")
    return result


def validate_data(data):
    """Require exactly 25 manufacturing orders with five readings and one average each."""
    errors = []
    measurements = data[data["measurement_type"] == "Single Measurement"].copy()
    averages = data[data["measurement_type"] == "Average"].copy()
    orders = data["manufacturing_order_number"].dropna().unique()
    if len(orders) != 25 or len(measurements) != 125 or len(averages) != 25 or len(data) != 150:
        errors.append(
            f"Expected 25 MOs / 125 measurements / 25 averages / 150 rows; "
            f"found {len(orders)} / {len(measurements)} / {len(averages)} / {len(data)}."
        )
    for order in sorted(orders):
        n = (measurements["manufacturing_order_number"] == order).sum()
        a = (averages["manufacturing_order_number"] == order).sum()
        if n != 5 or a != 1:
            errors.append(f"MO {order}: {n} measurements and {a} averages (expected 5 + 1).")
    for label, frame in [("measurements", measurements), ("averages", averages)]:
        if pd.to_numeric(frame["test_result"], errors="coerce").isna().any():
            errors.append(f"Invalid numeric values among {label}.")
    limits = data[["minimum_expected", "maximum_expected"]].apply(pd.to_numeric, errors="coerce")
    if limits.isna().any().any() or len(limits.drop_duplicates()) != 1:
        errors.append("Specification limits are missing, invalid, or inconsistent.")
    elif limits.iloc[0, 0] >= limits.iloc[0, 1]:
        errors.append("LSL must be lower than USL.")
    if measurements["registration_date"].isna().any():
        errors.append("Invalid measurement registration date.")
    if errors:
        raise ValueError("Data validation failed:\n - " + "\n - ".join(errors))
    print("PASS: 25 MOs; 125 measurements; 25 recorded averages.")
    return measurements, averages


def calculate_subgroups(measurements, averages):
    """Recompute subgroup means and ranges from the five measurements."""
    groups = measurements.groupby("manufacturing_order_number").agg(
        mean=("test_result", "mean"),
        range=("test_result", lambda v: v.max() - v.min()),
        date=("registration_date", "min"),
    ).reset_index()
    stored = averages[["manufacturing_order_number", "test_result"]].rename(
        columns={"test_result": "recorded_average"}
    )
    groups = groups.merge(stored, on="manufacturing_order_number", validate="one_to_one")
    groups["difference"] = groups["recorded_average"] - groups["mean"]
    groups = groups.sort_values(["date", "manufacturing_order_number"]).reset_index(drop=True)
    groups["subgroup"] = np.arange(1, len(groups) + 1)
    print(groups[["subgroup", "manufacturing_order_number", "mean", "range",
                  "recorded_average", "difference"]].to_string(index=False))
    if (groups["difference"].abs() > 0.005).any():
        print("WARNING: Recorded M3 averages differ from calculated means.")
        print("Statistical calculations use the individual measurements.")
    return groups


def validate_normality(measurements, groups, lsl, usl):
    """Check normality and create book-style histogram and probability plot."""
    values = measurements["test_result"].astype(float).to_numpy()
    mean = values.mean()
    overall_sigma = values.std(ddof=1)
    within_sigma = groups["range"].mean() / D2
    if overall_sigma <= 0 or within_sigma <= 0:
        raise ValueError("Standard deviations must be positive.")

    test = anderson(values, dist="norm")
    index = np.argmin(abs(np.asarray(test.significance_level) - 5.0))
    passed = bool(test.statistic < test.critical_values[index])
    print(f"Anderson-Darling statistic: {test.statistic:.4f}; "
          f"5% critical value: {test.critical_values[index]:.4f}")
    print("Normality not rejected." if passed else "Normality rejected.")

    # Histogram with normal curves for within-subgroup and overall variation.
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(values, bins=12, density=True, color="lightgray",
            edgecolor="black", label="Measurements")
    x = np.linspace(min(lsl, values.min()) - 0.1,
                    max(usl, values.max()) + 0.1, 500)
    ax.plot(x, norm.pdf(x, mean, within_sigma), "--", linewidth=2,
            label="Within")
    ax.plot(x, norm.pdf(x, mean, overall_sigma), "-", linewidth=2,
            label="Overall")
    ax.axvline(lsl, color="firebrick", linestyle=":", label="LSL")
    ax.axvline(usl, color="firebrick", linestyle=":", label="USL")
    ax.set(title="Individual measurements", xlabel="Value", ylabel="Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "histogram.png", dpi=150)
    plt.close(fig)

    # Normal probability paper: values on X, probabilities on Y.
    # Use normal quantiles to make probability ticks non-linear in percentage.
    ordered = np.sort(values)
    n = len(ordered)
    probabilities = (np.arange(1, n + 1) - 0.5) / n
    z = norm.ppf(probabilities)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(ordered, z, s=16, color="black")
    x_line = np.linspace(ordered.min() - 0.05, ordered.max() + 0.05, 200)
    ax.plot(x_line, (x_line - mean) / overall_sigma, color="gray", linewidth=1)
    probability_ticks = np.array([0.001, 0.01, 0.05, 0.20, 0.50,
                                  0.80, 0.95, 0.99, 0.999])
    ax.set_yticks(norm.ppf(probability_ticks))
    ax.set_yticklabels([".001", ".01", ".05", ".20", ".50",
                        ".80", ".95", ".99", ".999"])
    ax.set_ylim(norm.ppf(0.001), norm.ppf(0.999))
    ax.set(title="Normal Probability Plot", xlabel="Measurement",
           ylabel="Cumulative probability")
    ax.grid(alpha=0.2, linestyle=":")
    ax.text(0.97, 0.06, f"AD statistic: {test.statistic:.3f}",
            transform=ax.transAxes, ha="right")
    fig.tight_layout()
    fig.savefig(OUTPUT / "qq_plot.png", dpi=150)
    plt.close(fig)
    print("Updated histogram and normal probability plot saved to outputs/.")
    return passed


def validate_stability(groups):
    """Create X-bar and R charts, using constants for groups of five."""
    grand_mean = groups["mean"].mean()
    r_bar = groups["range"].mean()
    settings = [
        ("mean", "X-bar", grand_mean, grand_mean - A2 * r_bar,
         grand_mean + A2 * r_bar, "xbar.png"),
        ("range", "R", r_bar, D3 * r_bar, D4 * r_bar, "r_chart.png"),
    ]
    stable = True
    for field, label, center, lower, upper, name in settings:
        outside = (groups[field] < lower) | (groups[field] > upper)
        if outside.any():
            stable = False
            print(f"{label}: points beyond control limits: "
                  f"{groups.loc[outside, 'subgroup'].tolist()}")
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(groups["subgroup"], groups[field], "-o")
        for value, caption, style in [(center, "CL", "-"), (lower, "LCL", "--"), (upper, "UCL", "--")]:
            ax.axhline(value, ls=style, label=f"{caption}: {value:.4f}")
        ax.set(title=f"{label} control chart", xlabel="Subgroup", ylabel=label)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUTPUT / name, dpi=150)
        plt.close(fig)
    print("Control limits check passed." if stable else "WARNING: Process stability not confirmed.")
    print("First draft checks limit violations only (not run/trend rules).")
    return stable


def calculate_capability(measurements, groups, lsl, usl):
    """Cp/Cpk use within sigma; Pp/Ppk use total sample sigma."""
    values = measurements["test_result"].astype(float)
    mean = values.mean()
    r_bar = groups["range"].mean()
    within = r_bar / D2
    overall = values.std(ddof=1)
    if within <= 0 or overall <= 0:
        raise ValueError("Both standard deviations must be positive.")
    cp = (usl - lsl) / (6 * within)
    cpk = min((usl - mean) / (3 * within), (mean - lsl) / (3 * within))
    pp = (usl - lsl) / (6 * overall)
    ppk = min((usl - mean) / (3 * overall), (mean - lsl) / (3 * overall))
    print(f"Mean={mean:.5f}; R-bar={r_bar:.5f}; "
          f"Within sigma={within:.6f}; Overall sigma={overall:.6f}")
    print(f"Cp={cp:.3f}; Cpk={cpk:.3f}; Pp={pp:.3f}; Ppk={ppk:.3f}")
    return cp, cpk, pp, ppk


def main():
    """Simple menu that preserves progress between stages."""
    OUTPUT.mkdir(exist_ok=True)
    data = selected = measurements = averages = groups = None
    normality = stability = False
    while True:
        print("\n=== TEXLA CAPABILITY CALCULATION ===")
        print("1. Load Excel data")
        print("2. Select item and test")
        print("3. Validate 25 subgroups (125 measurements + 25 averages)")
        print("4. Review subgroup means and ranges")
        print("5. Normality: histogram, Q-Q and Anderson-Darling")
        print("6. Stability: X-bar and R charts")
        print("7. Calculate Cp, Cpk, Pp and Ppk")
        print("0. Exit")
        choice = input("Choice: ").strip()
        try:
            if choice == "1":
                filename = input("Excel path [data/capability_test.xlsx]: ").strip()
                data = load_data(filename or "data/capability_test.xlsx")
                selected = measurements = averages = groups = None
                normality = stability = False
                print(f"Loaded {len(data)} rows.")
                print(data[["item_number", "test"]].drop_duplicates().to_string(index=False))
            elif choice == "2":
                if data is None:
                    print("Complete step 1 first.")
                    continue
                item = input("Item [541893201BEN]: ").strip() or "541893201BEN"
                test = input("Test [TT0N22863]: ").strip() or "TT0N22863"
                selected = select_data(data, item, test)
                measurements = averages = groups = None
                normality = stability = False
                print(f"Selected {len(selected)} rows.")
            elif choice == "3":
                if selected is None:
                    print("Complete step 2 first.")
                    continue
                measurements = averages = groups = None
                measurements, averages = validate_data(selected)
            elif choice == "4":
                if measurements is None:
                    print("Complete step 3 first.")
                    continue
                groups = calculate_subgroups(measurements, averages)
            elif choice in ("5", "6", "7"):
                if groups is None:
                    print("Complete step 4 first.")
                    continue
                lsl = float(selected["minimum_expected"].iloc[0])
                usl = float(selected["maximum_expected"].iloc[0])
                if choice == "5":
                    normality = validate_normality(measurements, groups, lsl, usl)
                elif choice == "6":
                    stability = validate_stability(groups)
                else:
                    if not normality or not stability:
                        print("WARNING: Normality and/or stability have not been confirmed.")
                    calculate_capability(measurements, groups, lsl, usl)
            elif choice == "0":
                break
            else:
                print("Invalid option.")
        except (ValueError, FileNotFoundError, KeyError) as error:
            print(f"ERROR: {error}")


if __name__ == "__main__":
    main()
