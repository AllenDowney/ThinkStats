"""This file contains code for use with "Think Stats",
by Allen B. Downey, available from greenteapress.com

Copyright 2010 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from collections import defaultdict
import numpy as np
import pandas as pd
import thinkstats2


def read_fem_resp(dct_file="2002FemResp.dct", dat_file="2002FemResp.dat.gz", **options):
    """Reads the NSFG respondent data.

    dct_file: string file name
    dat_file: string file name

    returns: DataFrame
    """
    dct = thinkstats2.read_stata_dct(dct_file, encoding="iso-8859-1")
    df = dct.read_fixed_width(dat_file, compression="gzip", **options)
    return df


def read_fem_resp1995():
    """Reads respondent data from NSFG Cycle 5.

    returns: DataFrame
    """
    dat_file = "1995FemRespData.dat.gz"
    names = ["cmintvw", "timesmar", "cmmarrhx", "cmbirth", "finalwgt"]
    colspecs = [
        (12360 - 1, 12363),
        (4637 - 1, 4638),
        (11759 - 1, 11762),
        (14 - 1, 16),
        (12350 - 1, 12359),
    ]
    df = pd.read_fwf(dat_file, compression="gzip", colspecs=colspecs, names=names)
    df["timesmar"] = df.timesmar.replace([98, 99], np.nan)
    df["evrmarry"] = df.timesmar > 0
    clean_fem_resp(df)
    return df


def read_fem_resp2002():
    """Reads respondent data from NSFG Cycle 6.

    returns: DataFrame
    """
    usecols = [
        "caseid",
        "cmmarrhx",
        "cmdivorcx",
        "cmbirth",
        "cmintvw",
        "evrmarry",
        "parity",
        "finalwgt",
    ]
    df = read_fem_resp(usecols=usecols)
    df["evrmarry"] = df.evrmarry == 1
    clean_fem_resp(df)
    return df


def read_fem_resp2010():
    """Reads respondent data from NSFG Cycle 7.

    returns: DataFrame
    """
    usecols = [
        "caseid",
        "cmmarrhx",
        "cmdivorcx",
        "cmbirth",
        "cmintvw",
        "evrmarry",
        "parity",
        "wgtq1q16",
    ]
    df = read_fem_resp(
        "2006_2010_FemRespSetup.dct", "2006_2010_FemResp.dat.gz", usecols=usecols
    )
    df["evrmarry"] = df.evrmarry == 1
    df["finalwgt"] = df.wgtq1q16
    clean_fem_resp(df)
    return df


def read_fem_resp2013():
    """Reads respondent data from NSFG Cycle 8.

    returns: DataFrame
    """
    usecols = [
        "caseid",
        "cmmarrhx",
        "cmdivorcx",
        "cmbirth",
        "cmintvw",
        "evrmarry",
        "parity",
        "wgt2011_2013",
    ]
    df = read_fem_resp(
        "2011_2013_FemRespSetup.dct", "2011_2013_FemRespData.dat.gz", usecols=usecols
    )
    df["evrmarry"] = df.evrmarry == 1
    df["finalwgt"] = df.wgt2011_2013
    clean_fem_resp(df)
    return df


def clean_fem_resp(resp):
    """Cleans a respondent DataFrame.

    resp: DataFrame of respondents

    Adds columns: agemarry, age, decade, fives
    """
    resp["cmmarrhx"] = resp.cmmarrhx.replace([9997, 9998, 9999], np.nan)
    resp["agemarry"] = (resp.cmmarrhx - resp.cmbirth) / 12.0
    resp["age"] = (resp.cmintvw - resp.cmbirth) / 12.0
    month0 = pd.to_datetime("1899-12-15")
    dates = [(month0 + pd.DateOffset(months=cm)) for cm in resp.cmbirth]
    resp["year"] = pd.DatetimeIndex(dates).year - 1900
    resp["decade"] = resp.year // 10
    resp["fives"] = resp.year // 5


def read_fem_preg(dct_file="2002FemPreg.dct", dat_file="2002FemPreg.dat.gz"):
    """Reads the NSFG pregnancy data.

    dct_file: string file name
    dat_file: string file name

    returns: DataFrame
    """
    dct = thinkstats2.read_stata_dct(dct_file)
    df = dct.read_fixed_width(dat_file, compression="gzip")
    clean_fem_preg(df)
    return df


def clean_fem_preg(df):
    """Recodes variables from the pregnancy frame.

    df: DataFrame
    """
    df.agepreg /= 100.0
    df.loc[df.birthwgt_lb > 20, "birthwgt_lb"] = np.nan
    na_vals = [97, 98, 99]
    df["birthwgt_lb"] = df.birthwgt_lb.replace(na_vals, np.nan)
    df["birthwgt_oz"] = df.birthwgt_oz.replace(na_vals, np.nan)
    df["hpagelb"] = df.hpagelb.replace(na_vals, np.nan)
    df["babysex"] = df.babysex.replace([7, 9], np.nan)
    df["nbrnaliv"] = df.nbrnaliv.replace([9], np.nan)
    df["totalwgt_lb"] = df.birthwgt_lb + df.birthwgt_oz / 16.0
    df.cmintvw = np.nan


def validate_pregnum(resp, preg):
    """Validate pregnum in the respondent file.

    resp: respondent DataFrame
    preg: pregnancy DataFrame
    """
    preg_map = make_preg_map(preg)
    for index, pregnum in resp.pregnum.items():
        caseid = resp.caseid[index]
        indices = preg_map[caseid]
        if len(indices) != pregnum:
            print(caseid, len(indices), pregnum)
            return False
    return True


def make_preg_map(df):
    """Make a map from caseid to list of preg indices.

    df: DataFrame

    returns: dict that maps from caseid to list of indices into `preg`
    """
    d = defaultdict(list)
    for index, caseid in df.caseid.items():
        d[caseid].append(index)
    return d


def make_frames():
    """Reads pregnancy data and partitions first babies and others.

    returns: DataFrames (all live births, first babies, others)
    """
    preg = read_fem_preg()
    live = preg[preg.outcome == 1]
    firsts = live[live.birthord == 1]
    others = live[live.birthord != 1]
    assert len(live) == 9148
    assert len(firsts) == 4413
    assert len(others) == 4735
    return live, firsts, others


def summarize(live, firsts, others):
    """Print various summary statistics."""
    mean = live.prglngth.mean()
    var = live.prglngth.var()
    std = live.prglngth.std()
    print("Live mean", mean)
    print("Live variance", var)
    print("Live std", std)
    mean1 = firsts.prglngth.mean()
    mean2 = others.prglngth.mean()
    var1 = firsts.prglngth.var()
    var2 = others.prglngth.var()
    print("Mean")
    print("First babies", mean1)
    print("Others", mean2)
    print("Variance")
    print("First babies", var1)
    print("Others", var2)
    print("Difference in weeks", mean1 - mean2)
    print("Difference in hours", (mean1 - mean2) * 7 * 24)
    print("Difference relative to 39 weeks", (mean1 - mean2) / 39 * 100)
    d = thinkstats2.cohen_effect_size(firsts.prglngth, others.prglngth)
    print("Cohen d", d)


def main():
    """Tests the functions in this module.

    script: string script name
    """
    resp = read_fem_resp()
    assert len(resp) == 7643
    assert resp.pregnum.value_counts()[1] == 1267
    preg = read_fem_preg()
    print(preg.shape)
    assert len(preg) == 13593
    assert preg.caseid[13592] == 12571
    assert preg.pregordr.value_counts()[1] == 5033
    assert preg.nbrnaliv.value_counts()[1] == 8981
    assert preg.babysex.value_counts()[1] == 4641
    assert preg.birthwgt_lb.value_counts()[7] == 3049
    assert preg.birthwgt_oz.value_counts()[0] == 1037
    assert preg.prglngth.value_counts()[39] == 4744
    assert preg.outcome.value_counts()[1] == 9148
    assert preg.birthord.value_counts()[1] == 4413
    assert preg.agepreg.value_counts()[22.75] == 100
    assert preg.totalwgt_lb.value_counts()[7.5] == 302
    weights = preg.finalwgt.value_counts()
    key = max(weights.keys())
    assert preg.finalwgt.value_counts()[key] == 6
    assert validate_pregnum(resp, preg)
    print("All tests passed.")
    live, firsts, others = make_frames()
    summarize(live, firsts, others)


if __name__ == "__main__":
    main()
