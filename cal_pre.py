#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

H1 = pd.Timedelta(hours=1)
H4 = pd.Timedelta(hours=4)
H24 = pd.Timedelta(hours=24)

def _case(branches, index) -> pd.Series:
    conds = [c for c, _ in branches]
    vals = [v for _, v in branches]
    return pd.Series(np.select(conds, vals, default=np.nan), index=index)

def _between(s: pd.Series, lo: str, hi: str) -> pd.Series:
    return (s >= lo) & (s <= hi)

def _window_agg(co: pd.DataFrame, tbl: pd.DataFrame, key: str,
                aggs: dict[str, tuple[str, str]],
                chunk_size: int = 5000) -> pd.DataFrame:
    value_cols = sorted({c for c, _ in aggs.values()})
    tbl = tbl[[key, "cht", *value_cols]]
    base_cols = ["stay_id"] + ([key] if key != "stay_id" else []) + ["str", "edt"]
    base = co[base_cols]

    parts = []
    for i in range(0, len(base), chunk_size):
        m = base.iloc[i:i + chunk_size].merge(tbl, on=key, how="inner")
        m = m[(m["cht"] > m["str"]) & (m["cht"] <= m["edt"])]
        g = m.groupby("stay_id")
        out = {}
        for name, (col, fn) in aggs.items():
            if fn == "sum":
                out[name] = g[col].sum(min_count=1)  
            else:
                out[name] = getattr(g[col], fn)()
        parts.append(pd.DataFrame(out))
    res = pd.concat(parts) if parts else pd.DataFrame(columns=list(aggs))
    return res.reindex(co["stay_id"].values)

def build_cpap(co: pd.DataFrame, chartevents: pd.DataFrame) -> pd.DataFrame:
    ce = chartevents[chartevents["itemid"] == CPAP_ITEMID]
    ce = ce[ce["value"].str.lower().str.contains(CPAP_REGEX, na=False, regex=True)]

    m = co[["stay_id", "str", "edt"]].merge(
        ce[["stay_id", "cht"]], on="stay_id", how="inner")
    m = m[(m["cht"] > m["str"]) & (m["cht"] <= m["edt"])]

    g = m.groupby("stay_id")["cht"].agg(first="min", last="max").reset_index()
    cp = g.merge(co[["subject_id", "stay_id", "str", "edt"]], on="stay_id")
    cp["str"] = np.maximum(cp["first"] - H1, cp["str"])
    cp["edt"] = np.minimum(cp["last"] + H4, cp["edt"])
    cp["cpap"] = 1
    return cp[["subject_id", "stay_id", "str", "edt", "cpap"]]

def build_vt_min(co: pd.DataFrame, bg: pd.DataFrame,
                            vtl: pd.DataFrame,
                            cpap: pd.DataFrame) -> pd.Series:
    b = bg[bg["specimen"] == "ART."][["subject_id", "cht", "pao2fio2ratio"]]
    b = co[["subject_id", "stay_id", "str", "edt"]].merge(
        b, on="subject_id", how="inner")
    b = b[(b["cht"] > b["str"]) & (b["cht"] <= b["edt"])]
    b = b.reset_index(drop=True)
    b["rid"] = b.index

    v = (vtl[vtl["vtl_status"] == "InvasiveVent"]
         [["stay_id", "str", "edt"]]
         .rename(columns={"str": "v_start", "edt": "v_end"}))
    mv = b[["rid", "stay_id", "cht"]].merge(v, on="stay_id", how="inner")
    mv = mv[(mv["cht"] > mv["v_start"]) & (mv["cht"] <= mv["v_end"])]

    c = (cpap[["subject_id", "str", "edt"]]
         .rename(columns={"str": "c_start", "edt": "c_end"}))
    mc = b[["rid", "subject_id", "cht"]].merge(c, on="subject_id", how="inner")
    mc = mc[(mc["cht"] > mc["c_start"]) & (mc["cht"] <= mc["c_end"])]

    flagged = b["rid"].isin(mv["rid"]) | b["rid"].isin(mc["rid"])
    return b[flagged].groupby("stay_id")["pao2fio2ratio"].min()


def compute_sp(tables: dict[str, pd.DataFrame],
                   chunk_size: int = 5000) -> pd.DataFrame:
    t = {k: _prep(k, v) for k, v in tables.items()}

    co = build_co(t["wkstays"])
    cpap = build_cpap(co, t["chartevents"])
    surgflag = build_surgflag(t["admissions"], t["services"])
    comorb = build_comorb(t["diagnoses_icd"])
    pafi_min = build_pao2fio2_vent_min(co, t["bg"], t["vtl"], cpap)

    gcs = _window_agg(co, t["gcs"], "stay_id",
                      {"mingcs": ("gcs", "min")}, chunk_size)
    vital = _window_agg(co, t["vitalsign"], "subject_id", {
        "heartrate_min": ("heart_rate", "min"), "heartrate_max": ("heart_rate", "max"),
        "sysbp_min": ("sbp", "min"), "sysbp_max": ("sbp", "max"),
        "tempc_min": ("temperature", "min"), "tempc_max": ("temperature", "max"),
    }, chunk_size)
    uo = _window_agg(co, t["urine_output"], "stay_id",
                     {"urineoutput": ("urineoutput", "sum")}, chunk_size)
    labs = _window_agg(co, t["chemistry"], "subject_id", {
        "bun_min": ("bun", "min"), "bun_max": ("bun", "max"),
        "potassium_min": ("potassium", "min"), "potassium_max": ("potassium", "max"),
        "sodium_min": ("sodium", "min"), "sodium_max": ("sodium", "max"),
        "bicarbonate_min": ("bicarbonate", "min"),
        "bicarbonate_max": ("bicarbonate", "max"),
    }, chunk_size)
    cbc = _window_agg(co, t["complete_blood_count"], "subject_id", {
        "wbc_min": ("wbc", "min"), "wbc_max": ("wbc", "max"),
    }, chunk_size)
    enz = _window_agg(co, t["enzyme"], "subject_id", {
        "bilirubin_min": ("bilirubin_total", "min"),
        "bilirubin_max": ("bilirubin_total", "max"),
    }, chunk_size)

    ie = t["wkstays"][["subject_id", "hadm_id", "stay_id", "intime", "outtime"]]
    cohort = (ie
              .merge(t["admissions"][["hadm_id", "admission_type"]], on="hadm_id", how="inner")
              .merge(t["age"][["hadm_id", "age"]], on="hadm_id", how="left")
              .merge(co[["stay_id", "str", "edt"]], on="stay_id", how="inner")
              .merge(surgflag, on="hadm_id", how="left")
              .merge(comorb, on="hadm_id", how="left"))
    cohort["pao2fio2_vent_min"] = cohort["stay_id"].map(pafi_min)
    for df in (vital, uo, labs, cbc, enz, gcs):
        for col in df.columns:
            cohort[col] = cohort["stay_id"].map(df[col])  # df 的索引即 stay_id

    at, surg = cohort["admission_type"], cohort["surgical"]
    cohort["admissiontype"] = np.select(
        [(at == "ELECTIVE") & (surg == 1),
         at.notna() & (at != "ELECTIVE") & (surg == 1)],
        ["ScheduledSurgical", "UnscheduledSurgical"], default="Medical")

    c = cohort
    ix = c.index
    nan = np.nan

    age = c["age"]
    c["age_score"] = _case([
        (age.isna(), nan), (age < 40, 0), (age < 60, 7), (age < 70, 12),
        (age < 75, 15), (age < 80, 16), (age >= 80, 18)], ix)

    hmin, hmax = c["heartrate_min"], c["heartrate_max"]
    c["hr_score"] = _case([
        (hmax.isna(), nan), (hmin < 40, 11), (hmax >= 160, 7), (hmax >= 120, 4),
        (hmin < 70, 2),
        ((hmax >= 70) & (hmax < 120) & (hmin >= 70) & (hmin < 120), 0)], ix)

    smin, smax = c["sysbp_min"], c["sysbp_max"]
    c["sysbp_score"] = _case([
        (smin.isna(), nan), (smin < 70, 13), (smin < 100, 5), (smax >= 200, 2),
        ((smax >= 100) & (smax < 200) & (smin >= 100) & (smin < 200), 0)], ix)

    tmin, tmax = c["tempc_min"], c["tempc_max"]
    c["temp_score"] = _case([
        (tmax.isna(), nan), (tmax >= 39.0, 3), (tmin < 39.0, 0)], ix)

    pf = c["pao2fio2_vent_min"]
    c["pao2fio2_score"] = _case([
        (pf.isna(), nan), (pf < 100, 11), (pf < 200, 9), (pf >= 200, 6)], ix)

    u = c["urineoutput"]
    c["uo_score"] = _case([
        (u.isna(), nan), (u < 500.0, 11), (u < 1000.0, 4), (u >= 1000.0, 0)], ix)

    bun = c["bun_max"]
    c["bun_score"] = _case([
        (bun.isna(), nan), (bun < 28.0, 0), (bun < 84.0, 6), (bun >= 84.0, 10)], ix)

    wmin, wmax = c["wbc_min"], c["wbc_max"]
    c["wbc_score"] = _case([
        (wmax.isna(), nan), (wmin < 1.0, 12), (wmax >= 20.0, 3),
        ((wmax >= 1.0) & (wmax < 20.0) & (wmin >= 1.0) & (wmin < 20.0), 0)], ix)

    kmin, kmax = c["potassium_min"], c["potassium_max"]
    c["potassium_score"] = _case([
        (kmax.isna(), nan), (kmin < 3.0, 3), (kmax >= 5.0, 3),
        ((kmax >= 3.0) & (kmax < 5.0) & (kmin >= 3.0) & (kmin < 5.0), 0)], ix)

    nmin, nmax = c["sodium_min"], c["sodium_max"]
    c["sodium_score"] = _case([
        (nmax.isna(), nan), (nmin < 125, 5), (nmax >= 145, 1),
        ((nmax >= 125) & (nmax < 145) & (nmin >= 125) & (nmin < 145), 0)], ix)

    bmin, bmax = c["bicarbonate_min"], c["bicarbonate_max"]
    c["bicarbonate_score"] = _case([
        (bmax.isna(), nan), (bmin < 15.0, 6), (bmin < 20.0, 3),
        ((bmax >= 20.0) & (bmin >= 20.0), 0)], ix)

    bil = c["bilirubin_max"]
    c["bilirubin_score"] = _case([
        (bil.isna(), nan), (bil < 4.0, 0), (bil < 6.0, 4), (bil >= 6.0, 9)], ix)

    g = c["mingcs"]
    c["gcs_score"] = _case([
        (g.isna(), nan), (g < 3, nan), 
        (g < 6, 26), (g < 9, 13), (g < 11, 7), (g < 14, 5),
        ((g >= 14) & (g <= 15), 0)], ix)

    c["comorbidity_score"] = _case([
        (c["aids"] == 1, 17), (c["hem"] == 1, 10), (c["mets"] == 1, 9)], ix).fillna(0)

    adm = c["admissiontype"]
    c["admissiontype_score"] = _case([
        (adm == "ScheduledSurgical", 0), (adm == "Medical", 6),
        (adm == "UnscheduledSurgical", 8)], ix)

def _prep(name: str, df: pd.DataFrame) -> pd.DataFrame:
    spec = TABLE_SPECS[name]
    df = df[spec["cols"]].copy()
    for col in spec["times"]:
        df[col] = pd.to_datetime(df[col])
    for col in spec["cols"]:
        if col in spec["times"] or col in TEXT_COLS:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in spec["cols"]:
        if col in TEXT_COLS:
            df[col] = df[col].astype("object")
    return df


def _find_file(data_dir: Path, name: str) -> Path:
    for ext in (".csv.gz", ".csv", ".parquet"):
        hits = sorted(data_dir.rglob(f"{name}{ext}"))
        if hits:
            return hits[0]


def load_from_files(data_dir: str, max_stays: int | None = None
                    ) -> dict[str, pd.DataFrame]:
    root = Path(data_dir)
    tables: dict[str, pd.DataFrame] = {}
    subset = None

    order = ["wkstays"] + [k for k in TABLE_SPECS if k != "wkstays"]
    for name in order:
        spec = TABLE_SPECS[name]
        path = _find_file(root, name)
        print(f"[files] loading {path} ...", file=sys.stderr)
        cols = spec["cols"]

        def keep(df: pd.DataFrame) -> pd.DataFrame:
            if spec.get("where") and "itemid" in df.columns:
                df = df[df["itemid"] == CPAP_ITEMID]
            if subset is not None:
                df = df[df[spec["key"]].isin(subset[spec["key"]].unique())]
            return df

        if path.suffix == ".parquet":
            df = keep(pd.read_parquet(path, columns=cols))
        else:
            dtypes = {c: str for c in cols if c in TEXT_COLS}
            reader = pd.read_csv(path, usecols=cols, dtype=dtypes,
                                 parse_dates=spec["times"], chunksize=2_000_000)
            df = pd.concat((keep(chunk) for chunk in reader), ignore_index=True)

        if name == "wkstays" and max_stays:
            df = df.sort_values("stay_id").head(max_stays)
            subset = df
        tables[name] = df
    return tables

def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["bigquery", "csv"], required=True")
    p.add_argument("--project")
    p.add_argument("--data-dir")
    p.add_argument("--max-stays", type=int)
    p.add_argument("-o", "--output", default="sp.csv")
    args = p.parse_args(argv)

    if args.source == "bigquery":
        if not args.project:
            p.error("--source bigquery --project")
        subset = None
        tables = load(args.project, subset)
        if subset is not None:
            tables["wkstays"] = tables["wkstays"][
                tables["wkstays"]["stay_id"].isin(subset["stay_id"])]
    else:
        if not args.data_dir:
            p.error("--source csv --data-dir")
        tables = load_from_files(args.data_dir, args.max_stays)

    result = compute_sp(tables)
    if args.output.endswith(".parquet"):
        result.to_parquet(args.output, index=False)
    else:
        result.to_csv(args.output, index=False)
    print(f"Done: {len(result)} wk stays -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
