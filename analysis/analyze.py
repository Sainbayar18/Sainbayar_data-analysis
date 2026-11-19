"""
Dataset/ хавтас доторх 8 датасетийн шинжилгээ.

Ажиллуулах:  python analysis/analyze.py
Гаралт:      reports/figures/*.png  ба  reports/summary.json
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Dataset"
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font="DejaVu Sans")
plt.rcParams["figure.dpi"] = 110
SUMMARY: dict = {}


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)


def r(x, n=2):
    return None if pd.isna(x) else round(float(x), n)


# --------------------------------------------------------------------------
# 1. Лос Анжелес хотын гэмт хэрэг (2020 – одоо)
# --------------------------------------------------------------------------
def crime() -> None:
    path = DATA / "Crime_Data_from_2020_to_Present.csv"
    if not path.exists():
        print("! Crime dataset not found, skipping")
        return
    cols = ["DATE OCC", "TIME OCC", "AREA NAME", "Crm Cd Desc", "Vict Age",
            "Vict Sex", "Weapon Desc", "Status Desc", "Premis Desc"]
    df = pd.read_csv(path, usecols=cols, dtype={"TIME OCC": str})
    df["date"] = pd.to_datetime(df["DATE OCC"], format="%m/%d/%Y %I:%M:%S %p")
    df["hour"] = df["TIME OCC"].str.zfill(4).str[:2].astype(int)
    df["year"] = df["date"].dt.year

    monthly = df.set_index("date").resample("MS").size()
    # Сүүлийн 2 сар тайлагнал дутуу байж магадгүй тул графикт тэмдэглэнэ
    fig, ax = plt.subplots(figsize=(11, 4))
    monthly.plot(ax=ax, color="#2b6cb0")
    ax.set_title("LA гэмт хэрэг: сар бүрийн тоо")
    ax.set_xlabel("")
    ax.set_ylabel("Хэргийн тоо")
    save(fig, "crime_monthly.png")

    top = df["Crm Cd Desc"].value_counts().head(12)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(x=top.values, y=top.index, ax=ax, color="#c53030")
    ax.set_title("Хамгийн их бүртгэгдсэн 12 төрлийн гэмт хэрэг")
    ax.set_xlabel("Тоо")
    ax.set_ylabel("")
    save(fig, "crime_top_types.png")

    area = df["AREA NAME"].value_counts()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=area.values, y=area.index, ax=ax, color="#2f855a")
    ax.set_title("Бүс нутгаар (LAPD area)")
    ax.set_xlabel("Тоо")
    ax.set_ylabel("")
    save(fig, "crime_by_area.png")

    hourly = df["hour"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.bar(hourly.index, hourly.values, color="#6b46c1")
    ax.set_xticks(range(0, 24))
    ax.set_title("Өдрийн цагаар (TIME OCC)")
    ax.set_xlabel("Цаг")
    ax.set_ylabel("Тоо")
    save(fig, "crime_by_hour.png")

    ages = df.loc[(df["Vict Age"] > 0) & (df["Vict Age"] < 100), "Vict Age"]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.hist(ages, bins=50, color="#dd6b20")
    ax.set_title("Хохирогчийн нас (0 ба түүнээс доош утгыг хассан)")
    ax.set_xlabel("Нас")
    save(fig, "crime_victim_age.png")

    sex = df["Vict Sex"].map({"M": "Эрэгтэй", "F": "Эмэгтэй", "X": "Тодорхойгүй"}).value_counts(normalize=True)
    weapon_share = df["Weapon Desc"].notna().mean()
    SUMMARY["crime"] = {
        "rows": int(len(df)),
        "date_range": [str(df["date"].min().date()), str(df["date"].max().date())],
        "per_year": {int(k): int(v) for k, v in df["year"].value_counts().sort_index().items()},
        "top_types": {k: int(v) for k, v in top.head(5).items()},
        "top_areas": {k: int(v) for k, v in area.head(5).items()},
        "peak_hour": int(hourly.idxmax()),
        "noon_share_pct": r(100 * (df["hour"] == 12).mean(), 1),
        "victim_age_median": r(ages.median(), 0),
        "victim_age_zero_or_missing_pct": r(100 * (1 - len(ages) / len(df)), 1),
        "victim_sex_share_pct": {k: r(100 * v, 1) for k, v in sex.items()},
        "weapon_reported_pct": r(100 * weapon_share, 1),
        "status": {k: int(v) for k, v in df["Status Desc"].value_counts().items()},
        "top_premises": {k: int(v) for k, v in df["Premis Desc"].value_counts().head(5).items()},
    }


# --------------------------------------------------------------------------
# 2. Powerball сугалааны ялагч дугаарууд (2010 – )
# --------------------------------------------------------------------------
def powerball() -> None:
    df = pd.read_csv(DATA / "Lottery_Powerball_Winning_Numbers__Beginning_2010.csv",
                     parse_dates=["Draw Date"]).sort_values("Draw Date")
    nums = df["Winning Numbers"].str.split(expand=True).astype(int)
    df["white"] = nums.iloc[:, :5].values.tolist()
    df["pb"] = nums.iloc[:, 5]
    # 2015-10-07-оос эхлэн дүрэм өөрчлөгдсөн: цагаан 1-69, улаан 1-26
    cur = df[df["Draw Date"] >= "2015-10-07"]
    white = pd.Series(np.concatenate(cur["white"].values)).value_counts().reindex(range(1, 70), fill_value=0)
    pb = cur["pb"].value_counts().reindex(range(1, 27), fill_value=0)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    expected_w = white.sum() / 69
    axes[0].bar(white.index, white.values, color="#4a5568")
    axes[0].axhline(expected_w, color="red", ls="--", label=f"Хүлээгдэх ≈ {expected_w:.0f}")
    axes[0].set_title("Цагаан бөмбөг (1–69) гарсан давтамж, 2015-10-07-оос хойш")
    axes[0].legend()
    expected_p = pb.sum() / 26
    axes[1].bar(pb.index, pb.values, color="#c53030")
    axes[1].axhline(expected_p, color="black", ls="--", label=f"Хүлээгдэх ≈ {expected_p:.0f}")
    axes[1].set_title("Powerball (1–26) гарсан давтамж")
    axes[1].legend()
    save(fig, "powerball_frequency.png")

    # Chi-square тэгш тархалтын тест (scipy-гүйгээр статистик л тооцно)
    chi_w = float(((white - expected_w) ** 2 / expected_w).sum())
    chi_p = float(((pb - expected_p) ** 2 / expected_p).sum())

    SUMMARY["powerball"] = {
        "draws": int(len(df)),
        "date_range": [str(df["Draw Date"].min().date()), str(df["Draw Date"].max().date())],
        "draws_current_rules": int(len(cur)),
        "hot_white": {int(k): int(v) for k, v in white.sort_values(ascending=False).head(5).items()},
        "cold_white": {int(k): int(v) for k, v in white.sort_values().head(5).items()},
        "hot_pb": {int(k): int(v) for k, v in pb.sort_values(ascending=False).head(3).items()},
        "chi2_white_df68": round(chi_w, 1),
        "chi2_pb_df25": round(chi_p, 1),
        "multiplier_counts": {str(k): int(v) for k, v in df["Multiplier"].value_counts(dropna=False).items()},
    }


# --------------------------------------------------------------------------
# 3. Өрөөний хүний тоог мэдрэгчээр тооцох (Occupancy Estimation)
# --------------------------------------------------------------------------
def occupancy() -> None:
    df = pd.read_csv(DATA / "Occupancy_Estimation.csv")
    target = "Room_Occupancy_Count"
    feats = [c for c in df.columns if c not in ("Date", "Time", target)]
    corr = df[feats + [target]].corr()[target].drop(target).sort_values()

    fig, ax = plt.subplots(figsize=(8, 6))
    corr.plot.barh(ax=ax, color=np.where(corr > 0, "#2b6cb0", "#c53030"))
    ax.set_title("Мэдрэгч бүрийн хүний тоотой корреляци")
    save(fig, "occupancy_correlation.png")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col in zip(axes, ["S1_Light", "S5_CO2", "S1_Temp"]):
        sns.boxplot(data=df, x=target, y=col, ax=ax, color="#90cdf4")
        ax.set_xlabel("Хүний тоо")
    fig.suptitle("Хүний тоогоор мэдрэгчийн утга")
    save(fig, "occupancy_boxplots.png")

    # Энгийн босго дүрэм: S1_Light > босго → хүнтэй
    occupied = df[target] > 0
    best = max(((t, ((df["S1_Light"] > t) == occupied).mean()) for t in range(20, 400, 5)),
               key=lambda x: x[1])
    SUMMARY["occupancy"] = {
        "rows": int(len(df)),
        "dates": sorted(df["Date"].unique().tolist()),
        "class_counts": {int(k): int(v) for k, v in df[target].value_counts().sort_index().items()},
        "empty_room_pct": r(100 * (~occupied).mean(), 1),
        "top_corr": {k: r(v, 3) for k, v in corr.sort_values(ascending=False).head(5).items()},
        "light_threshold_rule": {"threshold": best[0], "accuracy_pct": r(100 * best[1], 1)},
        "mean_by_count": df.groupby(target)[["S1_Light", "S5_CO2", "S1_Temp"]].mean().round(1).to_dict(),
    }


# --------------------------------------------------------------------------
# 4. Мэдрэгчийн цагийн цуваа (Sensor_data)
# --------------------------------------------------------------------------
def sensor() -> None:
    df = pd.read_csv(DATA / "Sensor_data.csv", parse_dates=["time"]).set_index("time").sort_index()
    daily = df["val"].resample("D").agg(["mean", "min", "max", "count"])
    gaps = df.index.to_series().diff()
    big_gaps = gaps[gaps > pd.Timedelta("1h")]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(daily.index, daily["mean"], color="#2b6cb0", label="Өдрийн дундаж")
    ax.fill_between(daily.index, daily["min"], daily["max"], alpha=0.2, color="#2b6cb0", label="min–max")
    ax.set_title("Мэдрэгчийн утга (5 минут тутам), өдрөөр нэгтгэсэн")
    ax.legend()
    save(fig, "sensor_daily.png")

    by_hour = df.groupby(df.index.hour)["val"].mean()
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(by_hour.index, by_hour.values, marker="o", color="#6b46c1")
    ax.set_xticks(range(24))
    ax.set_title("Өдрийн цаг тутмын дундаж утга")
    save(fig, "sensor_by_hour.png")

    SUMMARY["sensor"] = {
        "rows": int(len(df)),
        "range": [str(df.index.min()), str(df.index.max())],
        "stats": {k: r(v) for k, v in df["val"].describe().items()},
        "share_at_50_pct": r(100 * (df["val"] == 50).mean(), 1),
        "gaps_over_1h": int(len(big_gaps)),
        "longest_gap": str(big_gaps.max()) if len(big_gaps) else None,
        "days_with_data": int((daily["count"] > 0).sum()),
        "calendar_days": int(len(daily)),
        "peak_hour": int(by_hour.idxmax()),
        "low_hour": int(by_hour.idxmin()),
    }


# --------------------------------------------------------------------------
# 5. Улаанбаатарын орон сууцны зар (unegui.mn)
# --------------------------------------------------------------------------
def apartments() -> None:
    df = pd.read_csv(DATA / "data_2.csv", encoding="utf-8-sig")
    df = df[(df["Үнэ"] > 5e6) & (df["Талбай"] >= 10) & (df["Талбай"] <= 200)].copy()
    df["price_m2"] = df["Үнэ"] / df["Талбай"]
    ub = df[~df["Дүүрэг"].isin(["Орон нутаг"])]
    main_d = ub["Дүүрэг"].value_counts()
    main_d = main_d[main_d >= 50].index
    ub = ub[ub["Дүүрэг"].isin(main_d)]

    order = ub.groupby("Дүүрэг")["price_m2"].median().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=ub, y="Дүүрэг", x=ub["price_m2"] / 1e6, order=order, ax=ax,
                showfliers=False, color="#90cdf4")
    ax.set_xlabel("1 м² үнэ (сая ₮)")
    ax.set_ylabel("")
    ax.set_title("Дүүргээр 1 м²-ын үнэ (outlier-гүй)")
    save(fig, "apartments_price_m2_by_district.png")

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(data=ub, x="Талбай", y=ub["Үнэ"] / 1e6, hue="Дүүрэг", s=10, alpha=0.5, ax=ax)
    ax.set_ylabel("Үнэ (сая ₮)")
    ax.set_xlabel("Талбай (м²)")
    ax.set_ylim(0, 400)
    ax.set_title("Талбай ба үнэ")
    ax.legend(fontsize=8, markerscale=2)
    save(fig, "apartments_area_vs_price.png")

    yr = ub[ub["Ашиглалтанд орсон он"] >= 1960]
    by_year = yr.groupby("Ашиглалтанд орсон он")["price_m2"].median() / 1e6
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(by_year.index, by_year.values, marker="o", ms=3, color="#2f855a")
    ax.set_title("Ашиглалтанд орсон оноор 1 м²-ын медиан үнэ")
    ax.set_ylabel("сая ₮ / м²")
    save(fig, "apartments_price_by_year.png")

    def med_by(col):
        g = ub.groupby(col)["price_m2"].agg(["median", "count"])
        g = g[g["count"] >= 30].sort_values("median", ascending=False)
        return {str(k): {"median_m2_mnt": int(v["median"]), "n": int(v["count"])} for k, v in g.iterrows()}

    corr_cols = ["Үнэ", "Талбай", "Ашиглалтанд орсон он", "Барилгын давхар", "Хэдэн давхарт", "Цонхны тоо"]
    SUMMARY["apartments"] = {
        "rows_after_cleaning": int(len(df)),
        "rows_ub_main_districts": int(len(ub)),
        "date_range": [str(df["Огноо"].min()), str(df["Огноо"].max())],
        "price_median_mnt": int(ub["Үнэ"].median()),
        "area_median_m2": r(ub["Талбай"].median(), 1),
        "price_m2_median_mnt": int(ub["price_m2"].median()),
        "by_district": med_by("Дүүрэг"),
        "by_leasing": med_by("Лизинг"),
        "by_floor_type": med_by("Шал"),
        "top_locations": med_by("Байршил") and dict(list(med_by("Байршил").items())[:8]),
        "corr_with_price": {k: r(v, 3) for k, v in ub[corr_cols].corr()["Үнэ"].drop("Үнэ").items()},
    }

    # Түүхий unegui.mn файлаас зөвхөн нэргүй үзүүлэлт (үзсэн тоо) авна.
    raw = pd.read_csv(DATA / "unegui.mn_dataset.csv", usecols=["Үзсэн", "Дүүрэг", "Үнэ"])
    SUMMARY["apartments"]["raw_rows"] = int(len(raw))
    SUMMARY["apartments"]["views_median"] = r(pd.to_numeric(raw["Үзсэн"], errors="coerce").median(), 0)


# --------------------------------------------------------------------------
# 6. Японы их, дээд сургуулиуд
# --------------------------------------------------------------------------
def japan() -> None:
    df = pd.read_csv(DATA / "japanese_universities.csv", index_col=0)
    df["found_year"] = pd.to_numeric(df["found"].astype(str).str[:4], errors="coerce")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.boxplot(data=df, x="type", y="difficulty_SD", ax=axes[0], order=["National", "Public", "Private"],
                color="#fbd38d")
    axes[0].set_title("Төрлөөр элсэлтийн хүндрэл (hensachi)")
    axes[0].set_xlabel("")
    sns.scatterplot(data=df, x="difficulty_SD", y="review_rating", hue="type", ax=axes[1], s=18, alpha=0.7)
    axes[1].set_title("Хүндрэл ба сурагчдын үнэлгээ")
    save(fig, "japan_difficulty.png")

    st = df["state"].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(x=st.values, y=st.index, ax=ax, color="#805ad5")
    ax.set_title("Муж (prefecture) бүрийн сургуулийн тоо — топ 15")
    ax.set_xlabel("Тоо")
    ax.set_ylabel("")
    save(fig, "japan_by_prefecture.png")

    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.hist(df["found_year"].dropna(), bins=range(1860, 2030, 5), color="#319795")
    ax.set_title("Байгуулагдсан он")
    save(fig, "japan_founded.png")

    top = df.sort_values("difficulty_SD", ascending=False).head(10)[["name", "type", "difficulty_SD", "review_rating"]]
    SUMMARY["japan"] = {
        "rows": int(len(df)),
        "by_type": {k: int(v) for k, v in df["type"].value_counts().items()},
        "difficulty_median_by_type": {k: r(v, 1) for k, v in df.groupby("type")["difficulty_SD"].median().items()},
        "rating_median_by_type": {k: r(v, 2) for k, v in df.groupby("type")["review_rating"].median().items()},
        "corr_difficulty_rating": r(df["difficulty_SD"].corr(df["review_rating"]), 3),
        "corr_difficulty_reviewcount": r(df["difficulty_SD"].corr(df["review_count"]), 3),
        "top_prefectures": {k: int(v) for k, v in st.head(5).items()},
        "tokyo_share_pct": r(100 * (df["state"] == st.index[0]).mean(), 1),
        "has_grad_pct": r(100 * df["has_grad"].mean(), 1),
        "founded_after_1945_pct": r(100 * (df["found_year"] >= 1945).mean(), 1),
        "top10_difficulty": top.to_dict(orient="records"),
    }


# --------------------------------------------------------------------------
# 7. Мэргэжлээр цалин (АНУ)
# --------------------------------------------------------------------------
def salaries() -> None:
    df = pd.read_csv(DATA / "salaries_by_college_major.csv").dropna(subset=["Starting Median Salary"])
    df["growth_pct"] = 100 * (df["Mid-Career Median Salary"] / df["Starting Median Salary"] - 1)
    df["spread"] = df["Mid-Career 90th Percentile Salary"] - df["Mid-Career 10th Percentile Salary"]
    d = df.sort_values("Mid-Career Median Salary")

    fig, ax = plt.subplots(figsize=(9, 11))
    ax.barh(d["Undergraduate Major"], d["Starting Median Salary"] / 1e3, color="#a0aec0", label="Эхлэх")
    ax.barh(d["Undergraduate Major"], (d["Mid-Career Median Salary"] - d["Starting Median Salary"]) / 1e3,
            left=d["Starting Median Salary"] / 1e3, color="#2b6cb0", label="Карьерын дунд үе хүртэлх өсөлт")
    ax.set_xlabel("Медиан цалин ($ мянга)")
    ax.set_title("Мэргэжлээр цалин: эхлэх ба карьерын дунд үе")
    ax.legend(loc="lower right")
    save(fig, "salaries_by_major.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=df, x="Group", y="Mid-Career Median Salary", ax=ax, color="#c6f6d5")
    ax.set_title("Бүлгээр карьерын дунд үеийн цалин")
    save(fig, "salaries_by_group.png")

    def top(col, n=5, asc=False):
        t = df.sort_values(col, ascending=asc).head(n)
        return {row["Undergraduate Major"]: r(row[col], 1) for _, row in t.iterrows()}

    SUMMARY["salaries"] = {
        "rows": int(len(df)),
        "top_starting": top("Starting Median Salary"),
        "top_midcareer": top("Mid-Career Median Salary"),
        "lowest_midcareer": top("Mid-Career Median Salary", asc=True),
        "top_growth_pct": top("growth_pct"),
        "widest_spread": top("spread"),
        "group_median_mid": {k: r(v, 0) for k, v in df.groupby("Group")["Mid-Career Median Salary"].median().items()},
    }


if __name__ == "__main__":
    for fn in (salaries, japan, powerball, occupancy, sensor, apartments, crime):
        print(f"→ {fn.__name__}")
        fn()
    out = ROOT / "reports" / "summary.json"
    out.write_text(json.dumps(SUMMARY, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"Done. {len(list(FIG.glob('*.png')))} figures, summary → {out}")
