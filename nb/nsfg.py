"""This file contains code for use with "Think Stats",
by Allen B. Downey, available from greenteapress.com

Copyright 2010 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

import numpy as np
import pandas as pd

from statadict import parse_stata_dict

from thinkstats import underride

def read_stata(dct_file, dat_file, **options):
    """Read data from a stata file.

    dct_file: string file name
    dat_file: string file name

    returns: DataFrame
    """
    stata_dict = parse_stata_dict(dct_file)

    underride(options, compression="gzip")
    resp = pd.read_fwf(
        dat_file,
        names=stata_dict.names,
        colspecs=stata_dict.colspecs,
        **options,
    )
    return resp


def read_fem_resp(dct_file="2002FemResp.dct", dat_file="2002FemResp.dat.gz"):
    """Read the 2002 NSFG respondent file.

    dct_file: string file name
    dat_file: string file name

    returns: DataFrame
    """
    resp =  read_stata(dct_file, dat_file)
    clean_fem_resp(resp)
    return resp


def read_fem_preg(dct_file="2002FemPreg.dct", dat_file="2002FemPreg.dat.gz"):
    """Reads the NSFG pregnancy data.

    dct_file: string file name
    dat_file: string file name

    returns: DataFrame
    """
    preg = read_stata(dct_file, dat_file)
    clean_fem_preg(preg)
    return preg


def get_nsfg_groups():
    """Read the NSFG pregnancy file and split into groups.
    
    returns: all live births, first babies, other babies
    """
    preg = read_fem_preg()
    live = preg.query("outcome == 1")
    firsts = live.query("birthord == 1")
    others = live.query("birthord != 1")
    return live, firsts, others


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


def clean_fem_preg(df):
    """Recodes variables from the pregnancy frame.

    df: DataFrame
    """
    df.agepreg /= 100.0
    df.loc[df.birthwgt_lb > 20, "birthwgt_lb"] = np.nan
    na_vals = [97, 98, 99]
    df["birthwgt_lb"] = df.birthwgt_lb.replace(na_vals, np.nan)
    df["birthwgt_oz"] = df.birthwgt_oz.replace(na_vals, np.nan)
    df["totalwgt_lb"] = df.birthwgt_lb + df.birthwgt_oz / 16.0

    df["hpagelb"] = df.hpagelb.replace(na_vals, np.nan)
    df["babysex"] = df.babysex.replace([7, 9], np.nan)
    df["nbrnaliv"] = df.nbrnaliv.replace([9], np.nan)


from statadict import parse_stata_dict


def read_variables():
    """Reads Stata dictionary files for NSFG data.

    returns: DataFrame that maps variables names to descriptions
    """
    vars1 = parse_stata_dict("2002FemPreg.dct").names
    vars2 = parse_stata_dict("2002FemResp.dct").names

    # TODO: update this to work with the new version of parse_stata_dict
    all_vars = np.concat([vars1, vars2])
    return all_vars


def join_fem_resp(df):
    """Reads the female respondent file and joins on caseid.

    df: DataFrame with caseid column

    returns: DataFrame
    """
    resp = read_fem_resp()
    resp.index = resp.caseid
    join = df.join(resp, on="caseid", rsuffix="_r")
    join.screentime = pd.to_datetime(join.screentime)
    return join



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
    print("All tests passed.")
    live, firsts, others = make_frames()


if __name__ == "__main__":
    main()
