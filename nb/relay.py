"""This file contains code used in "Think Stats",
by Allen B. Downey, available from greenteapress.com

Copyright 2014 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

import numpy as np
import thinkplot
import thinkstats2

"""
Sample line.

Place Div/Tot  Div   Guntime Nettime  Pace  Name                   Ag S Race# City/state              
===== ======== ===== ======= =======  ===== ====================== == = ===== ======================= 
   97  26/256  M4049   42:48   42:44   6:53 Allen Downey           42 M   337 Needham MA 
"""


def convert_pace_to_speed(pace):
    """Converts pace in MM:SS per mile to MPH."""
    m, s = [int(x) for x in pace.split(":")]
    secs = m * 60 + s
    mph = 1 / secs * 60 * 60
    return mph


def clean_line(line):
    """Converts a line from coolrunning results to a tuple of values."""
    t = line.split()
    if len(t) < 6:
        return None
    place, divtot, div, gun, net, pace = t[0:6]
    if not "/" in divtot:
        return None
    for time in [gun, net, pace]:
        if ":" not in time:
            return None
    return place, divtot, div, gun, net, pace


def read_results(filename="Apr25_27thAn_set1.shtml"):
    """Read results from a file and return a list of tuples."""
    results = []
    for line in open(filename):
        t = clean_line(line)
        if t:
            results.append(t)
    return results


def get_speeds(results, column=5):
    """Extract the pace column and return a list of speeds in MPH."""
    speeds = []
    for t in results:
        pace = t[column]
        speed = convert_pace_to_speed(pace)
        speeds.append(speed)
    return speeds


def bin_data(data, low, high, n):
    """Rounds data off into bins.

    data: sequence of numbers
    low: low value
    high: high value
    n: number of bins

    returns: sequence of numbers
    """
    data = (np.array(data) - low) / (high - low) * n
    data = np.round(data) * (high - low) / n + low
    return data


def main():
    results = read_results()
    speeds = get_speeds(results)
    speeds = bin_data(speeds, 3, 12, 100)
    pmf = thinkstats2.Pmf(speeds, "speeds")
    thinkplot.pmf(pmf)
    thinkplot.show(
        title="PMF of running speed", xlabel="speed (mph)", ylabel="probability"
    )


if __name__ == "__main__":
    main()
