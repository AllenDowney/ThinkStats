"""This file contains code used in "Think Stats",
by Allen B. Downey, available from greenteapress.com

Copyright 2010 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

import numpy as np
import pandas as pd
import thinkplot
import thinkstats2


def read_data(filename="PEP_2012_PEPANNRES_with_ann.csv"):
    """Reads filename and returns populations in thousands

    filename: string

    returns: pandas Series of populations in thousands
    """
    df = pd.read_csv(filename, header=None, skiprows=2, encoding="iso-8859-1")
    populations = df[7]
    populations.replace(0, np.nan, inplace=True)
    return populations.dropna()


def make_figures():
    """Plots the CDF of populations in several forms.

    On a log-log scale the tail of the CCDF looks like a straight line,
    which suggests a Pareto distribution, but that turns out to be misleading.

    On a log-x scale the distribution has the characteristic sigmoid of
    a lognormal distribution.

    The normal probability plot of log(sizes) confirms that the data fit the
    lognormal model very well.

    Many phenomena that have been described with Pareto models can be described
    as well, or better, with lognormal models.
    """
    pops = read_data()
    print("Number of cities/towns", len(pops))
    log_pops = np.log10(pops)
    cdf_log = thinkstats2.Cdf(log_pops, label="data")
    xs, ys = thinkstats2.render_pareto_cdf(xmin=5000, alpha=1.4, low=0, high=10000000.0)
    thinkplot.plot(np.log10(xs), 1 - ys, label="model", color="0.8")
    thinkplot.cdf(cdf_log, complement=True)
    thinkplot.config(xlabel="log10 population", ylabel="CCDF", yscale="log")
    thinkplot.save(root="populations_pareto")
    thinkplot.pre_plot(cols=2)
    mu, sigma = log_pops.mean(), log_pops.std()
    xs, ps = thinkstats2.render_normal_cdf(mu, sigma, low=0, high=8)
    thinkplot.plot(xs, ps, label="model", color="0.8")
    thinkplot.cdf(cdf_log)
    thinkplot.config(xlabel="log10 population", ylabel="CDF")
    thinkplot.sub_plot(2)
    thinkstats2.normal_probability_plot(log_pops, label="data")
    thinkplot.config(xlabel="z", ylabel="log10 population", xlim=[-5, 5])
    thinkplot.save(root="populations_normal")


def main():
    thinkstats2.random_seed(17)
    make_figures()


if __name__ == "__main__":
    main()
