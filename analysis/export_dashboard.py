"""
Dashboard-д зориулж Dataset/ доторх файлуудыг жижиг нэгтгэсэн JSON болгоно.

Ажиллуулах:  python analysis/export_dashboard.py
Гаралт:      dashboard/data/<id>.json   — бүр нь {rows, meta}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Dataset"
OUT = ROOT / "dashboard" / "data"
OUT.mkdir(parents=True, exist_ok=True)
CODE = Path(__file__).read_text(encoding="utf-8")
HEAD: list[dict] = []


def num(x, n=2):
    return None if pd.isna(x) else round(float(x), n)


def write(name: str, rows, inp: str) -> None:
    rows = json.loads(pd.DataFrame(rows).to_json(orient="records", force_ascii=False))
    meta = [{"label": "export_dashboard.py", "value": CODE, "type": "code", "lang": "python"},
            {"label": "Input", "value": inp}]
    (OUT / f"{name}.json").write_text(json.dumps({"rows": rows, "meta": meta}, ensure_ascii=False),
                                      encoding="utf-8")


def head(key, label, value, inp):
    HEAD.append({"key": key, "label": label, "value": num(value, 3), "input": inp})


# 1. Улаанбаатарын орон сууц ----------------------------------------------
f = "data_2.csv"
d = pd.read_csv(DATA / f, encoding="utf-8-sig")
d = d[(d["Үнэ"] > 5e6) & (d["Талбай"] >= 10) & (d["Талбай"] <= 200)].copy()
d["m2"] = d["Үнэ"] / d["Талбай"]
keep = d["Дүүрэг"].value_counts()
d = d[(d["Дүүрэг"] != "Орон нутаг") & d["Дүүрэг"].isin(keep[keep >= 50].index)]
g = d.groupby("Дүүрэг")
write("apt_district", pd.DataFrame({
    "district": g.size().index,
    "listings": g.size().values,
    "median_m2": g["m2"].median().round(0).values,
    "q1_m2": g["m2"].quantile(0.25).round(0).values,
    "q3_m2": g["m2"].quantile(0.75).round(0).values,
    "median_price": g["Үнэ"].median().round(0).values,
}).sort_values("median_m2", ascending=False), f)
loc = d.groupby("Байршил")["m2"].agg(["median", "size"])
loc = loc[loc["size"] >= 30].sort_values("median", ascending=False).head(10)
write("apt_locations", pd.DataFrame({"location": loc.index, "median_m2": loc["median"].round(0).values,
                                     "listings": loc["size"].values}), f)
yr = d[d["Ашиглалтанд орсон он"] >= 1960].groupby("Ашиглалтанд орсон он")["m2"].agg(["median", "size"])
yr = yr[yr["size"] >= 10]
write("apt_year", pd.DataFrame({"year": yr.index, "median_m2": yr["median"].round(0).values,
                                "listings": yr["size"].values}), f)
head("apt_n", "Шинжилсэн зар", len(d), f)
head("apt_price", "Медиан үнэ, ₮", d["Үнэ"].median(), f)
head("apt_area", "Медиан талбай, м²", d["Талбай"].median(), f)
head("apt_m2", "1 м²-ын медиан үнэ, ₮", d["m2"].median(), f)
head("apt_corr_area", "Талбай ба үнийн корреляци", d["Үнэ"].corr(d["Талбай"]), f)
head("apt_corr_year", "Ашиглалтын он ба үнийн корреляци", d["Үнэ"].corr(d["Ашиглалтанд орсон он"]), f)

# 2. Лос Анжелесийн гэмт хэрэг --------------------------------------------
f = "Crime_Data_from_2020_to_Present.csv"
c = pd.read_csv(DATA / f, usecols=["DATE OCC", "TIME OCC", "AREA NAME", "Crm Cd Desc", "Status Desc"],
                dtype={"TIME OCC": str})
c["date"] = pd.to_datetime(c["DATE OCC"], format="%m/%d/%Y %I:%M:%S %p")
m = c.set_index("date").resample("MS").size()
write("crime_monthly", pd.DataFrame({"month": m.index.strftime("%Y-%m-%d"), "crimes": m.values}), f)
t = c["Crm Cd Desc"].value_counts().head(10)
write("crime_types", pd.DataFrame({"type": t.index, "crimes": t.values}), f)
a = c["AREA NAME"].value_counts()
write("crime_areas", pd.DataFrame({"area": a.index, "crimes": a.values}), f)
h = c["TIME OCC"].str.zfill(4).str[:2].astype(int).value_counts().sort_index()
write("crime_hour", pd.DataFrame({"hour": h.index, "crimes": h.values}), f)
yrs = c["date"].dt.year.value_counts()
head("crime_total", "Нийт бүртгэгдсэн хэрэг", len(c), f)
head("crime_2022", "2022 оны хэрэг", yrs.get(2022, 0), f)
head("crime_2023", "2023 оны хэрэг", yrs.get(2023, 0), f)
head("crime_2024", "2024 оны хэрэг", yrs.get(2024, 0), f)
head("crime_arrest_pct", "Насанд хүрэгчийг баривчилсан, %", 100 * (c["Status Desc"] == "Adult Arrest").mean(), f)
head("crime_open_pct", "Мөрдөлт үргэлжилж буй, %", 100 * (c["Status Desc"] == "Invest Cont").mean(), f)

# 3. Powerball ----------------------------------------------------------------
f = "Lottery_Powerball_Winning_Numbers__Beginning_2010.csv"
p = pd.read_csv(DATA / f, parse_dates=["Draw Date"])
p = p[p["Draw Date"] >= "2015-10-07"]  # одоогийн дүрэм: цагаан 1–69, улаан 1–26
n = p["Winning Numbers"].str.split(expand=True).astype(int)
w = pd.Series(n.iloc[:, :5].values.ravel()).value_counts().reindex(range(1, 70), fill_value=0)
r = n.iloc[:, 5].value_counts().reindex(range(1, 27), fill_value=0)
ew, er = w.sum() / 69, r.sum() / 26
write("pb_white", pd.DataFrame({"ball": w.index, "draws": w.values, "expected": round(ew, 2)}), f)
write("pb_red", pd.DataFrame({"ball": r.index, "draws": r.values, "expected": round(er, 2)}), f)
head("pb_draws", "2015-10-07-оос хойших тохирол", len(p), f)
head("pb_chi_white", "χ² цагаан бөмбөг (df=68)", ((w - ew) ** 2 / ew).sum(), f)
head("pb_chi_red", "χ² Powerball (df=25)", ((r - er) ** 2 / er).sum(), f)

# 4. Өрөөний хүний тоо ---------------------------------------------------------
f = "Occupancy_Estimation.csv"
o = pd.read_csv(DATA / f)
y = "Room_Occupancy_Count"
feats = [x for x in o.columns if x not in ("Date", "Time", y)]
cr = o[feats + [y]].corr()[y].drop(y).sort_values(ascending=False)
write("occ_corr", pd.DataFrame({"sensor": cr.index, "corr": cr.round(3).values}), f)
gm = o.groupby(y)
write("occ_by_count", pd.DataFrame({"people": gm.size().index, "readings": gm.size().values,
                                    "light": gm["S1_Light"].mean().round(1).values,
                                    "co2": gm["S5_CO2"].mean().round(0).values,
                                    "temp": gm["S1_Temp"].mean().round(2).values}), f)
occ = o[y] > 0
best = max(((th, ((o["S1_Light"] > th) == occ).mean()) for th in range(20, 400, 5)), key=lambda z: z[1])
head("occ_rows", "Хэмжилт", len(o), f)
head("occ_empty_pct", "Өрөө хоосон байсан хугацаа, %", 100 * (~occ).mean(), f)
head("occ_rule_th", "Гэрлийн босго (S1_Light)", best[0], f)
head("occ_rule_acc", "Босго дүрмийн нарийвчлал, %", 100 * best[1], f)

# 5. Мэдрэгчийн цагийн цуваа ----------------------------------------------------
f = "Sensor_data.csv"
s = pd.read_csv(DATA / f, parse_dates=["time"]).set_index("time").sort_index()
dl = s["val"].resample("D").agg(["mean", "min", "max"]).dropna()
write("sensor_daily", pd.DataFrame({"day": dl.index.strftime("%Y-%m-%d"), "mean": dl["mean"].round(2).values,
                                    "min": dl["min"].values, "max": dl["max"].values}), f)
gaps = s.index.to_series().diff()
head("sensor_rows", "Хэмжилт", len(s), f)
head("sensor_floor_pct", "Яг 50 утгатай хэмжилт, %", 100 * (s["val"] == 50).mean(), f)
head("sensor_days", "Өгөгдөлтэй өдөр", len(dl), f)
head("sensor_span", "Хамрах хугацаа, өдөр", (s.index.max().normalize() - s.index.min().normalize()).days + 1, f)
head("sensor_gap", "Хамгийн урт тасалдал, өдөр", gaps.max() / pd.Timedelta("1D"), f)

# 6. Японы их сургуулиуд ----------------------------------------------------
f = "japanese_universities.csv"
j = pd.read_csv(DATA / f, index_col=0)
write("jp_unis", j[["name", "type", "state", "difficulty_SD", "review_rating", "review_count"]]
      .dropna(subset=["difficulty_SD", "review_rating"]), f)
gj = j.groupby("type")
write("jp_type", pd.DataFrame({"type": gj.size().index, "universities": gj.size().values,
                               "difficulty_median": gj["difficulty_SD"].median().round(1).values,
                               "rating_median": gj["review_rating"].median().round(2).values})
      .sort_values("universities", ascending=False), f)
head("jp_n", "Сургууль", len(j), f)
head("jp_private_pct", "Хувийн сургууль, %", 100 * (j["type"] == "Private").mean(), f)
head("jp_tokyo", "Токио дахь сургууль", (j["state"] == "Tokyo To").sum(), f)
head("jp_corr", "Хүндрэл ба үнэлгээний корреляци", j["difficulty_SD"].corr(j["review_rating"]), f)

# 7. Мэргэжлээр цалин ---------------------------------------------------------
f = "salaries_by_college_major.csv"
sl = pd.read_csv(DATA / f).dropna(subset=["Starting Median Salary"])
write("salaries", pd.DataFrame({
    "major": sl["Undergraduate Major"], "group": sl["Group"],
    "start": sl["Starting Median Salary"], "mid": sl["Mid-Career Median Salary"],
    "p10": sl["Mid-Career 10th Percentile Salary"], "p90": sl["Mid-Career 90th Percentile Salary"],
    "growth_pct": (100 * (sl["Mid-Career Median Salary"] / sl["Starting Median Salary"] - 1)).round(1),
}).sort_values("mid", ascending=False), f)
for grp in ("STEM", "Business", "HASS"):
    head(f"sal_{grp.lower()}", f"{grp} медиан цалин (карьерын дунд үе), $",
         sl.loc[sl["Group"] == grp, "Mid-Career Median Salary"].median(), f)

write("headline", HEAD, "Dataset/ доторх 7 файл (дээрх кодыг үз)")
print("wrote", sorted(p.name for p in OUT.glob("*.json")))
