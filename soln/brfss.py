"""This file contains code for use with "Think Stats",
by Allen B. Downey, available from greenteapress.com

Copyright 2010 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

import math
import sys
import pandas as pd
import numpy as np
import thinkstats2
import thinkplot


def summarize(df, column, title):
    """Print summary statistics male, female and all."""
    items = [
        ("all", df[column]),
        ("male", df[df.sex == 1][column]),
        ("female", df[df.sex == 2][column]),
    ]
    print(title)
    print("key\tn\tmean\tvar\tstd\tcv")
    for key, series in items:
        mean, var = series.mean(), series.var()
        std = math.sqrt(var)
        cv = std / mean
        t = key, len(series), mean, var, std, cv
        print("%s\t%d\t%4.2f\t%4.2f\t%4.2f\t%4.4f" % t)


def clean_brfss_frame(df):
    """Recodes BRFSS variables.

    df: DataFrame
    """
    df["age"] = df.age.replace([7, 9], float("NaN"))
    df["htm3"] = df.htm3.replace([999], float("NaN"))
    df["wtkg2"] = df.wtkg2.replace([99999], float("NaN"))
    df.wtkg2 /= 100.0
    df["wtyrago"] = df.wtyrago.replace([7777, 9999], float("NaN"))
    df["wtyrago"] = df.wtyrago.apply(lambda x: x / 2.2 if x < 9000 else x - 9000)


def read_brfss(filename="CDBRFS08.ASC.gz", compression="gzip", nrows=None):
    """Reads the BRFSS data.

    filename: string
    compression: string
    nrows: int number of rows to read, or None for all

    returns: DataFrame
    """
    var_info = [
        ("age", 101, 102, int),
        ("sex", 143, 143, int),
        ("wtyrago", 127, 130, int),
        ("finalwt", 799, 808, int),
        ("wtkg2", 1254, 1258, int),
        ("htm3", 1251, 1253, int),
    ]
    columns = ["name", "start", "end", "type"]
    variables = pd.DataFrame(var_info, columns=columns)
    variables.end += 1
    dct = thinkstats2.FixedWidthVariables(variables, index_base=1)
    df = dct.read_fixed_width(filename, compression=compression, nrows=nrows)
    clean_brfss_frame(df)
    return df


def make_normal_model(weights):
    """Plots a CDF with a Normal model.

    weights: sequence
    """
    cdf = thinkstats2.Cdf(weights, label="weights")
    mean, var = thinkstats2.trimmed_mean_var(weights)
    std = math.sqrt(var)
    print("n, mean, std", len(weights), mean, std)
    xmin = mean - 4 * std
    xmax = mean + 4 * std
    xs, ps = thinkstats2.render_normal_cdf(mean, std, xmin, xmax)
    thinkplot.plot(xs, ps, label="model", linewidth=4, color="0.8")
    thinkplot.cdf(cdf)


def make_normal_plot(weights):
    """Generates a normal probability plot of birth weights.

    weights: sequence
    """
    mean, var = thinkstats2.trimmed_mean_var(weights, p=0.01)
    std = math.sqrt(var)
    xs = [-5, 5]
    xs, ys = thinkstats2.fit_line(xs, mean, std)
    thinkplot.plot(xs, ys, color="0.8", label="model")
    xs, ys = thinkstats2.normal_probability(weights)
    thinkplot.plot(xs, ys, label="weights")


def make_figures(df):
    """Generates CDFs and normal prob plots for weights and log weights."""
    weights = df.wtkg2.dropna()
    log_weights = np.log10(weights)
    thinkplot.pre_plot(cols=2)
    make_normal_model(weights)
    thinkplot.config(xlabel="adult weight (kg)", ylabel="CDF")
    thinkplot.sub_plot(2)
    make_normal_model(log_weights)
    thinkplot.config(xlabel="adult weight (log10 kg)")
    thinkplot.save(root="brfss_weight")
    thinkplot.pre_plot(cols=2)
    make_normal_plot(weights)
    thinkplot.config(xlabel="z", ylabel="weights (kg)")
    thinkplot.sub_plot(2)
    make_normal_plot(log_weights)
    thinkplot.config(xlabel="z", ylabel="weights (log10 kg)")
    thinkplot.save(root="brfss_weight_normal")


def main(script, nrows=1000):
    """Tests the functions in this module.

    script: string script name
    """
    thinkstats2.random_seed(17)
    nrows = int(nrows)
    df = read_brfss(nrows=nrows)
    make_figures(df)
    summarize(df, "htm3", "Height (cm):")
    summarize(df, "wtkg2", "Weight (kg):")
    summarize(df, "wtyrago", "Weight year ago (kg):")
    if nrows == 1000:
        assert df.age.value_counts()[40] == 28
        assert df.sex.value_counts()[2] == 668
        assert df.wtkg2.value_counts()[90.91] == 49
        assert df.wtyrago.value_counts()[160 / 2.2] == 49
        assert df.htm3.value_counts()[163] == 103
        assert df.finalwt.value_counts()[185.870345] == 13
        print("%s: All tests passed." % script)


if __name__ == "__main__":
    main(*sys.argv)
