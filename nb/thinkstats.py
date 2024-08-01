"""This file contains code for use with "Think Stats", 3rd edition
by Allen B. Downey, available from greenteapress.com

Copyright 2024 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html

"""

import bisect
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy
from statadict import parse_stata_dict

from empiricaldist import Pmf, Cdf

from scipy.stats import norm


# Make the figures smaller to save some screen real estate.
# The figures generated for the book have DPI 400, so scaling
# them by a factor of 4 restores them to the size in the notebooks.
plt.rcParams['figure.dpi'] = 75
plt.rcParams['figure.figsize'] = [6, 3.5]


## Chapter 1

def read_stata(dct_file, dat_file, **options):
    """Read NSFG data.

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

## Chapter 2

def two_bar_plots(dist1, dist2, width=0.45, xlabel="", **options):
    """Makes two back-to-back bar plots.

    dist1: Hist or Pmf object
    dist2: Hist or Pmf object
    width: width of the bars
    options: passed along to plt.bar
    """
    underride(options, alpha=0.6)
    dist1.bar(align="edge", width=-width, **options)
    dist2.bar(align="edge", width=width, **options)
    decorate(xlabel=xlabel)


def cohen_effect_size(group1, group2):
    """Computes Cohen's effect size for two groups.

    group1: Series
    group2: Series

    returns: float
    """
    diff = group1.mean() - group2.mean()

    v1, v2 = group1.var(), group2.var()
    n1, n2 = group1.count(), group2.count()
    pooled_var = (n1 * v1 + n2 * v2) / (n1 + n2)

    return diff / np.sqrt(pooled_var)


## Chapter 3



## Chapter 5

def read_brfss(filename="CDBRFS08.ASC.gz", compression="gzip", nrows=None):
    """Reads the BRFSS data.

    filename: string
    compression: string
    nrows: int number of rows to read, or None for all

    returns: DataFrame
    """
    var_info = [
        ("age", 100, 102, int),
        ("sex", 142, 143, int),
        ("wtyrago", 126, 130, int),
        ("finalwt", 798, 808, int),
        ("wtkg2", 1253, 1258, int),
        ("htm3", 1250, 1253, int),
    ]
    columns = ["name", "start", "end", "type"]
    variables = pd.DataFrame(var_info, columns=columns)
    # variables["end"] += 1

    colspecs = variables[["start", "end"]].values.tolist()
    names = variables["name"].tolist()

    df = pd.read_fwf(filename,
                     colspecs=colspecs,
                     names=names,
                     compression=compression,
                     nrows=nrows)

    clean_brfss(df)
    return df


def clean_brfss(df):
    """Recodes BRFSS variables.

    df: DataFrame
    """
    df["age"] = df["age"].replace([7, 9], np.nan)
    df["htm3"] = df["htm3"].replace([999], np.nan)
    df["wtkg2"] = df["wtkg2"].replace([99999], np.nan) / 100
    df["wtyrago"] = df.wtyrago.replace([7777, 9999], np.nan)
    df["wtyrago"] = df.wtyrago.apply(lambda x: x / 2.2 if x < 9000 else x - 9000)

from scipy.special import comb


def binomial_pmf(k, n, p):
    """Compute the binomial PMF.

    k (int or array-like): number of successes
    n (int): number of trials
    p (float): probability of success on a single trial

    returns: float or ndarray
    """
    return comb(n, k) * (p**k) * ((1 - p) ** (n - k))


from scipy.special import factorial


def poisson_pmf(k, lam):
    """Compute the Poisson PMF.

    k (int or array-like): The number of occurrences
    lam (float): The rate parameter (λ) of the Poisson distribution

    returns: float or ndarray
    """
    return (lam**k) * np.exp(-lam) / factorial(k)


def exponential_cdf(x, lam):
    """Compute the exponential CDF.

    x: float or sequence of floats
    lam: rate parameter
    
    returns: float or NumPy array of cumulative probability
    """
    return 1 - np.exp(-lam * x)


def make_normal_model(data):
    """Make the Cdf of a normal distribution based on data.

    data: sequence of numbers
    """
    m, s = np.mean(data), np.std(data)
    low, high = np.min(data), np.max(data)
    qs = np.linspace(low, high, 201)
    ps = norm.cdf(qs, m, s)
    return Cdf(ps, qs, name="normal model")


def two_cdf_plots(cdf_model, cdf_data, xlabel="", **options):
    """Plot an empirical CDF and a theoretical model.

    cdf_model: Cdf object
    cdf_data: Cdf object
    xlabel: string
    options: control the way cdf_data is plotted
    """
    cdf_model.plot(alpha=0.6, color="gray")
    cdf_data.plot(alpha=0.6, **options)

    decorate(xlabel=xlabel, ylabel="CDF")

# Chapter 6

def normal_pdf(xs, mu, sigma):
    """Evaluates the normal probability density function.

    xs: float or sequence of floats
    mu: mean of the distribution
    sigma: standard deviation of the distribution

    returns: float or NumPy array of probability density
    """
    z = (xs - mu) / sigma
    return np.exp(-(z**2) / 2) / sigma / np.sqrt(2 * np.pi)


class Density:
    """Represents a continuous PDF or CDF."""

    def __init__(self, density_func, domain, name=""):
        """Initializes the Pdf.

        density_func: density function
        label: string
        """
        self.name = name
        self.density_func = density_func
        self.domain = domain

    def __repr__(self):
        return f"Density({self.density_func.__name__}, {self.domain}, name={self.name})"

    def __call__(self, qs):
        """Evaluates this Pdf at qs.

        qs: float or sequence of floats
        
        returns: float or NumPy array of probability density
        """
        return self.density_func(qs)
    
    def plot(self, qs=None, **options):
        """Plots this Pdf.

        qs: NumPy array of quantities where the Pdf should be evaluated
        options: passed along to plt.plot
        """
        if qs is None:
            low, high = self.domain
            qs = np.linspace(low, high, 201)
        ps = self(qs)
        underride(options, label=self.name)
        plt.plot(qs, ps, **options)


class Pdf(Density):
    """Represents a PDF."""

    def make_pmf(self, qs=None, **options):
        """Makes a discrete approximation to the Pdf.

        qs: NumPy array of quantities where the Pdf should be evaluated
        options: passed along to the Pmf constructor

        returns: Pmf
        """
        if qs is None:
            low, high = self.domain
            qs = np.linspace(low, high, 201)
        ps = self(qs)

        underride(options, name=self.name)
        pmf = Pmf(ps, qs, **options)
        pmf.normalize()
        return pmf
    

class ContinuousCdf(Density):
    """Represents a CDF."""

    def make_cdf(self, qs=None, **options):
        """Makes a discrete approximation to the CDF.

        qs: NumPy array of quantities where the CDF should be evaluated
        options: passed along to the Cdf constructor

        returns: Cdf
        """
        if qs is None:
            low, high = self.domain
            qs = np.linspace(low, high, 201)
        ps = self(qs)

        underride(options, name=self.name)
        cdf = Cdf(ps, qs, **options)
        return cdf
    

class NormalPdf(Pdf):
    """Represents the PDF of a Normal distribution."""

    def __init__(self, mu=0, sigma=1, domain=None, name=""):
        """Constructs a NormalPdf with given mu and sigma.

        mu: mean
        sigma: standard deviation
        name: string
        """
        self.mu = mu
        self.sigma = sigma
        if domain is None:
            domain = mu - 4 * sigma, mu + 4 * sigma
        self.domain = domain
        self.name = name

    def __repr__(self):
        """Returns a string representation."""
        return f"NormalPdf({self.mu}, {self.sigma}, name='{self.name}')"

    def __call__(self, qs):
        """Evaluates this PDF at qs.

        qs: scalar or sequence of floats

        returns: float or NumPy array of probability density
        """
        return normal_pdf(qs, self.mu, self.sigma)


class NormalCdf(ContinuousCdf):
    """Represents the CDF of a Normal distribution."""

    def __init__(self, mu=0, sigma=1, domain=None, name=""):
        """Constructs a NormalCdf with given mu and sigma.

        mu: mean
        sigma: standard deviation
        name: string
        """
        self.mu = mu
        self.sigma = sigma
        if domain is None:
            domain = mu - 4 * sigma, mu + 4 * sigma
        self.domain = domain
        self.name = name

    def __repr__(self):
        """Returns a string representation."""
        return f"NormalCdf({self.mu}, {self.sigma}, name='{self.name}')"
    
    def __call__(self, qs):
        """Evaluates this CDF at qs.

        qs: scalar or sequence of floats

        returns: float or NumPy array of cumulative probability
        """
        return norm.cdf(qs, self.mu, self.sigma)
    

def exponential_pdf(x, lam):
    """Evaluates the exponential PDF.

    x: float or sequence of floats
    lam: rate parameter

    returns: float or NumPy array of probability density
    """
    return lam * np.exp(-lam * x)


class ExponentialPdf(Pdf):
    """Represents the PDF of an exponential distribution."""

    def __init__(self, lam=1, domain=None, name=""):
        """Constructs an ExponentialPdf with given lambda.

        lam: rate parameter
        name: string
        """
        self.lam = lam
        if domain is None:
            domain = 0, 5.0 / lam
        self.domain = domain
        self.name = name

    def __repr__(self):
        """Returns a string representation."""
        return f"ExponentialPdf({self.lam}, name='{self.name}')"
    
    def __call__(self, qs):
        """Evaluates this PDF at qs.

        qs: scalar or sequence of floats

        returns: float or NumPy array of probability density
        """
        return exponential_pdf(qs, self.lam)
    

class ExponentialCdf(ContinuousCdf):
    """Represents the CDF of an exponential distribution."""

    def __init__(self, lam=1, domain=None, name=""):
        """Constructs an ExponentialCdf with given lambda.

        lam: rate parameter
        name: string
        """
        self.lam = lam
        if domain is None:
            domain = 0, 5.0 / lam
        self.domain = domain
        self.name = name

    def __repr__(self):
        """Returns a string representation."""
        return f"ExponentialCdf({self.lam}, name='{self.name}')"
    
    def __call__(self, qs):
        """Evaluates this CDF at qs.

        qs: scalar or sequence of floats

        returns: float or NumPy array of cumulative probability
        """
        return exponential_cdf(qs, self.lam)


def read_baby_boom(filename="babyboom.dat"):
    """Reads the babyboom data.

    filename: string

    returns: DataFrame
    """
    colspecs = [(1, 8), (9, 16), (17, 24), (25, 32)]
    column_names = ["time", "sex", "weight_g", "minutes"]
    df = pd.read_fwf(filename, colspecs=colspecs, names=column_names, skiprows=59)
    return df


## Chapter 7

def jitter(seq, std=1):
    """Jitters the values by adding random Gaussian noise.

    seq: sequence of numbers
    std: standard deviation of the added noise

    returns: new Numpy array
    """
    n = len(seq)
    return np.random.normal(0, std, n) + seq


def scatter(df, var1, var2, jitter_std=None, **options):
    """Make a scatter plot and return the coefficient of correlation.

    df: DataFrame
    var1: string variable name
    var2: string variable name
    jitter_std: float standard deviation of noise to add
    **options: passed along to plt.scatter
    """
    valid = df.dropna(subset=[var1, var2])
    xs = valid[var1]
    ys = valid[var2]

    if jitter_std is not None:
        xs = jitter(xs, jitter_std)
        ys = jitter(ys, jitter_std)

    underride(options, s=5, alpha=0.2)
    plt.scatter(xs, ys, **options)


def decile_plot(df, var1, var2, **options):
    """Make a decile plot.

    df: DataFrame
    var1: string variable name
    var2: string variable name
    **options: passed along to plt.plot
    """
    valid = df.dropna(subset=[var1, var2])
    deciles = pd.qcut(valid[var1], 10, labels=False)
    df_groupby = valid.groupby(deciles)
    series_groupby = df_groupby[var2]

    low = series_groupby.quantile(0.1)
    median = series_groupby.quantile(0.5)
    high = series_groupby.quantile(0.9)

    xs = df_groupby[var1].median()

    plt.fill_between(xs, low, high, alpha=0.2)
    underride(options, color="C0", label='median')
    plt.plot(xs, median, **options)


def corrcoef(df, var1, var2):
    """Computes the correlation matrix for two variables.

    df: DataFrame
    var1: string variable name
    var2: string variable name

    returns: float
    """
    valid = df.dropna(subset=[var1, var2])
    xs = valid[var1]
    ys = valid[var2]
    return np.corrcoef(xs, ys)[0, 1]


def rankcorr(df, var1, var2):
    """Computes the Spearman rank correlation for two variables.

    df: DataFrame
    var1: string variable name
    var2: string variable name

    returns: float
    """
    valid = df.dropna(subset=[var1, var2])
    xs = valid[var1].rank()
    ys = valid[var2].rank()
    return np.corrcoef(xs, ys)[0, 1]


def standardize(xs):
    """Standardizes a sequence of numbers.

    xs: sequence of numbers

    returns: NumPy array
    """
    return (xs - np.mean(xs)) / np.std(xs)


def make_correlated_scatter(xs, ys, rho, **options):
    """Makes a scatter plot with given correlation.
    
    xs: sequence of values
    ys: sequence of values
    rho: target correlation
    """
    ys = rho * xs + np.sqrt(1 - rho**2) * ys

    underride(options, s=5, alpha=0.5)
    plt.scatter(xs, ys, **options)
    add_rho(rho)
    remove_spines()


def add_rho(rho):
    """Adds a label to a figure to indicate the correlation."""
    ax = plt.gca()
    plt.text(0.5, 0.05, f"ρ = {rho}",
        fontsize="large",
        transform=ax.transAxes,
        ha="center",
        va="center",
    )


def make_nonlinear_scatter(xs, ys, kind="quadratic", **options):
    """Makes a scatter plot with a nonlinear relationship.

    xs: sequence of values
    ys: sequence of values
    """
    if kind == "quadratic":
        ys = ys + xs**2
    elif kind == "sinusoid":
        ys = ys + 10 * np.sin(3 * xs)
    elif kind == "abs":
        ys = ys / 4 - np.abs(xs)

    underride(options, s=5, alpha=0.5)
    plt.scatter(xs, ys, **options)
    remove_spines()
    r = np.corrcoef(xs, ys)[0, 1]
    return r


def remove_spines():
    """Remove the spines from a plot."""
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(False)

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_ticks([])


def cov(xs, ys):
    """Covariance of two variables.

    xs: sequence of values
    ys: sequence of values

    returns: float
    """
    xbar = np.mean(xs)
    ybar = np.mean(ys)
    dx = xs - xbar
    dy = ys - ybar
    cov = np.mean(dx * dy)
    return cov

def corr(xs, ys):
    """Correlation coefficient for two variables.

    xs: sequence of values
    ys: sequence of values

    returns: float
    """
    sx = np.std(xs)
    sy = np.std(ys)
    corr = cov(xs, ys) / sx / sy
    return corr


## Chapter 12

def percentile_rows(row_seq, percentiles):
    """Generates a sequence of percentiles from a sequence of rows.

    row_seq: sequence of rows
    percentiles: sequence of percentiles

    returns: sequence of percentiles
    """
    array = np.asarray(row_seq)
    return np.percentile(array, percentiles, axis=0)


## Chapter 14








def predict(xs, inter, slope):
    """Predicted values of y for given xs.

    xs: sequence of x
    inter: float intercept
    slope: float slope

    returns: sequence of y
    """
    xs = np.asarray(xs)
    return inter + slope * xs


def fit_line(xs, inter, slope):
    """Fits a line to the given data.

    xs: sequence of x
    inter: float intercept
    slope: float slope

    returns: sequence of x, sequence of y
    """
    low, high = np.min(xs), np.max(xs)
    fit_xs = np.linspace(low, high)
    fit_ys = predict(fit_xs, inter, slope)
    return fit_xs, fit_ys


def odds(p):
    """Computes odds for a given probability.

    Example: p=0.75 means 75 for and 25 against, or 3:1 odds in favor.

    Note: when p=1, the formula for odds divides by zero, which is
    normally undefined.  But I think it is reasonable to define Odds(1)
    to be infinity, so that's what this function does.

    p: float 0-1

    Returns: float odds
    """
    if p == 1:
        return float("inf")
    return p / (1 - p)


def probability(o):
    """Computes the probability corresponding to given odds.

    Example: o=2 means 2:1 odds in favor, or 2/3 probability

    o: float odds, strictly positive

    Returns: float probability
    """
    return o / (o + 1)


def probability2(yes, no):
    """Computes the probability corresponding to given odds.

    Example: yes=2, no=1 means 2:1 odds in favor, or 2/3 probability.

    yes, no: int or float odds in favor
    """
    return yes / (yes + no)


def confidence_interval(cdf, percent=90):
    """Compute a confidence interval.

    cdf: Cdf object
    percent: percent to be included

    returns: Numpy array
    """
    alpha = 1 - percent / 100
    return cdf.inverse([alpha / 2, 1 - alpha / 2])





class Interpolator(object):
    """Represents a mapping between sorted sequences; performs linear interp.

    Attributes:
        xs: sorted list
        ys: sorted list
    """

    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

    def lookup(self, x):
        """Looks up x and returns the corresponding value of y."""
        return self._Bisect(x, self.xs, self.ys)

    def reverse(self, y):
        """Looks up y and returns the corresponding value of x."""
        return self._Bisect(y, self.ys, self.xs)

    def _Bisect(self, x, xs, ys):
        """Helper function."""
        if x <= xs[0]:
            return ys[0]
        if x >= xs[-1]:
            return ys[-1]
        i = bisect.bisect(xs, x)
        frac = 1.0 * (x - xs[i - 1]) / (xs[i] - xs[i - 1])
        y = ys[i - 1] + frac * 1.0 * (ys[i] - ys[i - 1])
        return y


def make_uniform_pmf(low, high, n):
    """Make a uniform Pmf.

    low: lowest value (inclusive)
    high: highest value (inclusize)
    n: number of values
    """
    pmf = Pmf()
    for x in np.linspace(low, high, n):
        pmf.set(x, 1)
    pmf.normalize()
    return pmf



def resample(xs, n=None):
    """Draw a sample from xs with the same length as xs.

    xs: sequence
    n: sample size (default: len(xs))

    returns: NumPy array
    """
    if n is None:
        n = len(xs)
    return np.random.choice(xs, n, replace=True)


def sample_rows(df, n, replace=False):
    """Choose a sample of rows from a DataFrame.

    df: DataFrame
    n: number of rows
    replace: whether to sample with replacement

    returns: DataFrame
    """
    return df.sample(n, replace=replace)


def resample_rows(df):
    """Resamples rows from a DataFrame.

    df: DataFrame

    returns: DataFrame
    """
    n = len(df)
    return df.sample(n, replace=True)


def resample_rows_weighted(df, column="finalwgt"):
    """Resamples a DataFrame using probabilities proportional to given column.

    df: DataFrame
    column: string column name to use as weights

    returns: DataFrame
    """
    n = len(df)
    weights = df[column]
    return df.sample(n, weights=weights, replace=True)


class HypothesisTest(object):
    """Represents a hypothesis test."""

    def __init__(self, data):
        """Initializes.

        data: data in whatever form is relevant
        """
        self.data = data
        self.make_model()
        self.actual = self.test_statistic(data)
        self.test_stats = None
        self.test_cdf = None

    def p_value(self, iters=1000):
        """Computes the distribution of the test statistic and p-value.

        iters: number of iterations

        returns: float p-value
        """
        self.test_stats = [self.test_statistic(self.run_model()) for _ in range(iters)]
        self.test_cdf = Cdf.from_seq(self.test_stats)
        count = sum(1 for x in self.test_stats if x >= self.actual)
        return count / iters

    def max_test_stat(self):
        """Returns the largest test statistic seen during simulations."""
        return max(self.test_stats)

    def plot_cdf(self, label=None):
        """Draws a Cdf with vertical lines at the observed test stat."""

        def vert_line(x):
            """Draws a vertical line at x."""
            plt.plot([x, x], [0, 1], color="0.8")

        vert_line(self.actual)
        self.test_cdf.plot(label=label)

    def test_statistic(self, data):
        """Computes the test statistic.

        data: data in whatever form is relevant
        """
        raise NotImplementedError()

    def make_model(self):
        """Build a model of the null"""
        pass

    def run_model(self):
        """Run the model of the null

        returns: simulated data
        """
        raise NotImplementedError()


class CoinTest(HypothesisTest):
    """Tests the hypothesis that a coin is fair."""

    def test_statistic(self, data):
        """Computes the test statistic.

        data: data in whatever form is relevant
        """
        heads, tails = data
        test_stat = abs(heads - tails)
        return test_stat

    def run_model(self):
        """Run the model of the null

        returns: simulated data
        """
        heads, tails = self.data
        n = heads + tails
        sample = [random.choice("HT") for _ in range(n)]
        hist = Hist.from_seq(sample)
        data = hist["H"], hist["T"]
        return data


class DiffMeansPermute(HypothesisTest):
    """Tests a difference in means by permutation."""

    def test_statistic(self, data):
        """Computes the test statistic.

        data: data in whatever form is relevant
        """
        group1, group2 = data
        test_stat = abs(group1.mean() - group2.mean())
        return test_stat

    def make_model(self):
        """Build a model of the null"""
        group1, group2 = self.data
        self.n, self.m = len(group1), len(group2)
        self.pool = np.hstack((group1, group2))

    def run_model(self):
        """Run the model of the null

        returns: simulated data
        """
        np.random.shuffle(self.pool)
        data = self.pool[: self.n], self.pool[self.n :]
        return data


class DiffMeansOneSided(DiffMeansPermute):
    """Tests a one-sided difference in means by permutation."""

    def test_statistic(self, data):
        """Computes the test statistic.

        data: data in whatever form is relevant
        """
        group1, group2 = data
        test_stat = group1.mean() - group2.mean()
        return test_stat


class DiffStdPermute(DiffMeansPermute):
    """Tests a one-sided difference in standard deviation by permutation."""

    def test_statistic(self, data):
        """Computes the test statistic.

        data: data in whatever form is relevant
        """
        group1, group2 = data
        test_stat = group1.std() - group2.std()
        return test_stat


class CorrelationPermute(HypothesisTest):
    """Tests correlations by permutation."""

    def test_statistic(self, data):
        """Computes the test statistic.

        data: tuple of xs and ys
        """
        xs, ys = data
        test_stat = abs(corr(xs, ys))
        return test_stat

    def run_model(self):
        """Run the model of the null

        returns: simulated data
        """
        xs, ys = self.data
        xs = np.random.permutation(xs)
        return xs, ys


class DiceTest(HypothesisTest):
    """Tests whether a six-sided die is fair."""

    def test_statistic(self, data):
        """Computes the test statistic.

        data: list of frequencies
        """
        observed = data
        n = sum(observed)
        expected = np.ones(6) * n / 6
        test_stat = sum(abs(observed - expected))
        return test_stat

    def run_model(self):
        """Run the model of the null

        returns: simulated data
        """
        n = sum(self.data)
        values = [1, 2, 3, 4, 5, 6]
        rolls = np.random.choice(values, n, replace=True)
        hist = Hist.from_seq(rolls)
        freqs = hist(values)
        return freqs


class DiceChiTest(DiceTest):
    """Tests a six-sided die using a chi-squared statistic."""

    def test_statistic(self, data):
        """Computes the test statistic.

        data: list of frequencies
        """
        observed = data
        n = sum(observed)
        expected = np.ones(6) * n / 6
        test_stat = sum((observed - expected) ** 2 / expected)
        return test_stat


class PregLengthTest(HypothesisTest):
    """Tests difference in pregnancy length using a chi-squared statistic."""

    def test_statistic(self, data):
        """Computes the test statistic.

        data: pair of lists of pregnancy lengths
        """
        firsts, others = data
        stat = self.chi_squared(firsts) + self.chi_squared(others)
        return stat

    def chi_squared(self, lengths):
        """Computes the chi-squared statistic.

        lengths: sequence of lengths

        returns: float
        """
        hist = Hist.from_seq(lengths)
        observed = hist(self.values)
        expected = self.expected_probs * len(lengths)
        stat = sum((observed - expected) ** 2 / expected)
        return stat

    def make_model(self):
        """Build a model of the null"""
        firsts, others = self.data
        self.n = len(firsts)
        self.pool = np.hstack((firsts, others))
        pmf = Pmf(self.pool)
        self.values = range(35, 44)
        self.expected_probs = np.array(pmf.probs(self.values))

    def run_model(self):
        """Run the model of the null

        returns: simulated data
        """
        np.random.shuffle(self.pool)
        data = self.pool[: self.n], self.pool[self.n :]
        return data





def summarize_results(results):
    """Prints the most important parts of linear regression results:

    results: RegressionResults object
    """
    for name, param in results.params.items():
        pvalue = results.pvalues[name]
        print("%s   %0.3g   (%.3g)" % (name, param, pvalue))
    try:
        print("R^2 %.4g" % results.rsquared)
        ys = results.model.endog
        print("Std(ys) %.4g" % ys.std())
        print("Std(res) %.4g" % results.resid.std())
    except AttributeError:
        print("R^2 %.4g" % results.prsquared)


def print_tabular(rows, header):
    """Prints results in LaTeX tabular format.

    rows: list of rows
    header: list of strings
    """
    s = "\\hline " + " & ".join(header) + " \\\\ \\hline"
    print(s)
    for row in rows:
        s = " & ".join(row) + " \\\\"
        print(s)
    print("\\hline")


class Normal(object):
    """Represents a Normal distribution"""

    def __init__(self, mu, sigma2, label=""):
        """Initializes.

        mu: mean
        sigma2: variance
        """
        self.mu = mu
        self.sigma2 = sigma2
        self.label = label

    def __repr__(self):
        """Returns a string representation."""
        if self.label:
            return "Normal(%g, %g, %s)" % (self.mu, self.sigma2, self.label)
        else:
            return "Normal(%g, %g)" % (self.mu, self.sigma2)

    __str__ = __repr__

    @property
    def sigma(self):
        """Returns the standard deviation."""
        return np.sqrt(self.sigma2)

    def __add__(self, other):
        """Adds a number or other Normal.

        other: number or Normal

        returns: new Normal
        """
        if isinstance(other, Normal):
            return Normal(self.mu + other.mu, self.sigma2 + other.sigma2)
        else:
            return Normal(self.mu + other, self.sigma2)

    __radd__ = __add__

    def __sub__(self, other):
        """Subtracts a number or other Normal.

        other: number or Normal

        returns: new Normal
        """
        if isinstance(other, Normal):
            return Normal(self.mu - other.mu, self.sigma2 + other.sigma2)
        else:
            return Normal(self.mu - other, self.sigma2)

    __rsub__ = __sub__

    def __mul__(self, factor):
        """Multiplies by a scalar.

        factor: number

        returns: new Normal
        """
        return Normal(factor * self.mu, factor**2 * self.sigma2)

    __rmul__ = __mul__

    def __div__(self, divisor):
        """Divides by a scalar.

        divisor: number

        returns: new Normal
        """
        return 1.0 / divisor * self

    __truediv__ = __div__

    def sum(self, n):
        """Returns the distribution of the sum of n values.

        n: int

        returns: new Normal
        """
        return Normal(n * self.mu, n * self.sigma2)

    def plot(self, n_sigmas=4, **options):
        """Returns pair of xs, ys suitable for plotting.

        n_sigmas: how many sigmas to show
        options: passed along to plot
        """
        underride(options, label=self.label)
        mu, sigma = self.mu, self.sigma
        low, high = mu - n_sigmas * sigma, mu + 3 * sigma
        xs, ys = render_normal_cdf(mu, sigma, low, high)
        plt.plot(xs, ys, **options)

    def prob(self, x):
        """Returns the CDF of x.

        x: float

        returns: float probability
        """
        return scipy.stats.norm.cdf(x, self.mu, self.sigma)

    def percentile(self, p):
        """Computes a percentile of a normal distribution.

        p: float or sequence 0-100

        returns: float or array
        """
        return scipy.stats.norm.ppf(p/100, self.mu, self.sigma)


def student_cdf(n):
    """Discrete approximation of the CDF of Student's t distribution.

    n: sample size

    returns: Cdf
    """
    ts = np.linspace(-3, 3, 101)
    ps = scipy.stats.t.cdf(ts, df=n - 2)
    rs = ts / np.sqrt(n - 2 + ts**2)
    return Cdf(rs, ps)


def chi_squared_cdf(n):
    """Discrete approximation of the chi-squared CDF with df=n-1.

    n: sample size

    returns: Cdf
    """
    xs = np.linspace(0, 25, 101)
    ps = scipy.stats.chi2.cdf(xs, df=n - 1)
    return Cdf(ps, xs)




##  Plotting functions


def underride(d, **options):
    """Add key-value pairs to d only if key is not in d.

    d: dictionary
    options: keyword args to add to d
    """
    for key, val in options.items():
        d.setdefault(key, val)

    return d


def decorate(**options):
    """Decorate the current axes.
    Call decorate with keyword arguments like
    decorate(title='Title',
             xlabel='x',
             ylabel='y')
    The keyword arguments can be any of the axis properties
    https://matplotlib.org/api/axes_api.html
    In addition, you can use `legend=False` to suppress the legend.
    And you can use `loc` to indicate the location of the legend
    (the default value is 'best')
    """
    loc = options.pop("loc", "best")
    if options.pop("legend", True):
        legend(loc=loc)

    plt.gca().set(**options)
    plt.tight_layout()


def legend(**options):
    """Draws a legend only if there is at least one labeled item.
    options are passed to plt.legend()
    https://matplotlib.org/api/_as_gen/matplotlib.pyplot.legend.html
    """
    underride(options, loc="best")

    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, **options)
