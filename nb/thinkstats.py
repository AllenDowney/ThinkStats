"""This file contains code for use with "Think Stats"
by Allen B. Downey, available from greenteapress.com

Copyright 2024 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html


This file contains class definitions for:

_DictWrapper: private parent class for Hist and Pmf.

Hist: represents a histogram (map from values to integer frequencies).

Pmf: represents a probability mass function (map from values to probs).

Cdf: represents a discrete cumulative distribution function

Pdf: represents a continuous probability density function

"""

import bisect
import copy
import logging
import math
import random
import re
from collections import Counter
from io import open
from operator import itemgetter
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy
import thinkplot
from scipy import ndimage, special, stats
from scipy.special import gamma
import statsmodels.formula.api as smf

from empiricaldist import Pmf, Cdf

def random_seed(x):
    """Initialize the random and np.random generators.

    x: int seed
    """
    random.seed(x)
    np.random.seed(x)


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




class _DictWrapper(object):
    """An object that contains a dictionary."""

    def __init__(self, obj=None, name=None):
        """Initializes the distribution.

        obj: Hist, Pmf, Cdf, Pdf, dict, pandas Series, list of pairs
        label: string label
        """
        self.name = name if name is not None else ""
        self.d = {}
        self.log = False
        if obj is None:
            return
        if isinstance(obj, (_DictWrapper, Cdf, Pdf)):
            self.label = name if name is not None else obj.name
        if isinstance(obj, dict):
            self.d.update(obj.items())
        elif isinstance(obj, (_DictWrapper, Cdf, Pdf)):
            self.d.update(obj.items())
        elif isinstance(obj, pd.Series):
            self.d.update(obj.value_counts().items())
        else:
            self.d.update(Counter(obj))
        if len(self) > 0 and isinstance(self, Pmf):
            self.normalize()

    def __hash__(self):
        return id(self)

    def __str__(self):
        cls = self.__class__.__name__
        if self.label == "":
            return "%s(%s)" % (cls, str(self.d))
        else:
            return self.label

    def __repr__(self):
        cls = self.__class__.__name__
        if self.label == "":
            return "%s(%s)" % (cls, repr(self.d))
        else:
            return "%s(%s, %s)" % (cls, repr(self.d), repr(self.label))

    def __eq__(self, other):
        try:
            return self.d == other.d
        except AttributeError:
            return False

    def __len__(self):
        return len(self.d)

    def __iter__(self):
        return iter(self.d)

    def iterkeys(self):
        """Returns an iterator over keys."""
        return iter(self.d)

    def __contains__(self, value):
        return value in self.d

    def __getitem__(self, value):
        return self.d.get(value, 0)

    def __setitem__(self, value, prob):
        self.d[value] = prob

    def __delitem__(self, value):
        del self.d[value]

    def copy(self, label=None):
        """Returns a copy.

        Make a shallow copy of d.  If you want a deep copy of d,
        use copy.deepcopy on the whole object.

        label: string label for the new Hist

        returns: new _DictWrapper with the same type
        """
        new = copy.copy(self)
        new.d = copy.copy(self.d)
        new.label = label if label is not None else self.label
        return new

    def scale(self, factor):
        """Multiplies the values by a factor.

        factor: what to multiply by

        Returns: new object
        """
        new = self.copy()
        new.d.clear()
        for val, prob in self.items():
            new.set(val * factor, prob)
        return new

    def log(self, m=None):
        """Log transforms the probabilities.

        Removes values with probability 0.

        Normalizes so that the largest logprob is 0.
        """
        if self.log:
            raise ValueError("Pmf/Hist already under a log transform")
        self.log = True
        if m is None:
            m = self.max_like()
        for x, p in self.d.items():
            if p:
                self.set(x, math.log(p / m))
            else:
                self.remove(x)

    def exp(self, m=None):
        """Exponentiates the probabilities.

        m: how much to shift the ps before exponentiating

        If m is None, normalizes so that the largest prob is 1.
        """
        if not self.log:
            raise ValueError("Pmf/Hist not under a log transform")
        self.log = False
        if m is None:
            m = self.max_like()
        for x, p in self.d.items():
            self.set(x, math.exp(p - m))

    def get_dict(self):
        """Gets the dictionary."""
        return self.d

    def set_dict(self, d):
        """Sets the dictionary."""
        self.d = d

    def values(self):
        """Gets an unsorted sequence of values.

        Note: one source of confusion is that the keys of this
        dictionary are the values of the Hist/Pmf, and the
        values of the dictionary are frequencies/probabilities.
        """
        return self.d.keys()

    def items(self):
        """Gets an unsorted sequence of (value, freq/prob) pairs."""
        return self.d.items()

    def sorted_items(self):
        """Gets a sorted sequence of (value, freq/prob) pairs.

        It items are unsortable, the result is unsorted.
        """

        def isnan(x):
            try:
                return math.isnan(x)
            except TypeError:
                return False

        if any([isnan(x) for x in self.values()]):
            msg = "Keys contain NaN, may not sort correctly."
            logging.warning(msg)
        try:
            return sorted(self.d.items())
        except TypeError:
            return self.d.items()

    def render(self, **options):
        """Generates a sequence of points suitable for plotting.

        Note: options are ignored

        Returns:
            tuple of (sorted value sequence, freq/prob sequence)
        """
        return zip(*self.sorted_items())

    def make_cdf(self, label=None):
        """Makes a Cdf."""
        label = label if label is not None else self.label
        return Cdf(self, label=label)

    def print(self):
        """Prints the values and freqs/probs in ascending order."""
        for val, prob in self.sorted_items():
            print(val, prob)

    def set(self, x, y=0):
        """Sets the freq/prob associated with the value x.

        Args:
            x: number value
            y: number freq or prob
        """
        self.d[x] = y

    def incr(self, x, term=1):
        """Increments the freq/prob associated with the value x.

        Args:
            x: number value
            term: how much to increment by
        """
        self.d[x] = self.d.get(x, 0) + term

    def mult(self, x, factor):
        """Scales the freq/prob associated with the value x.

        Args:
            x: number value
            factor: how much to multiply by
        """
        self.d[x] = self.d.get(x, 0) * factor

    def remove(self, x):
        """Removes a value.

        Throws an exception if the value is not there.

        Args:
            x: value to remove
        """
        del self.d[x]

    def total(self):
        """Returns the total of the frequencies/probabilities in the map."""
        total = sum(self.d.values())
        return total

    def max_like(self):
        """Returns the largest frequency/probability in the map."""
        return max(self.d.values())

    def largest(self, n=10):
        """Returns the largest n values, with frequency/probability.

        n: number of items to return
        """
        return sorted(self.d.items(), reverse=True)[:n]

    def smallest(self, n=10):
        """Returns the smallest n values, with frequency/probability.

        n: number of items to return
        """
        return sorted(self.d.items(), reverse=False)[:n]


class Hist(_DictWrapper):
    """Represents a histogram, which is a map from values to frequencies.

    Values can be any hashable type; frequencies are integer counters.
    """

    def freq(self, x):
        """Gets the frequency associated with the value x.

        Args:
            x: number value

        Returns:
            int frequency
        """
        return self.d.get(x, 0)

    def freqs(self, xs):
        """Gets frequencies for a sequence of values."""
        return [self.freq(x) for x in xs]

    def is_subset(self, other):
        """Checks whether the values in this histogram are a subset of
        the values in the given histogram."""
        for val, freq in self.items():
            if freq > other.freq(val):
                return False
        return True

    def subtract(self, other):
        """Subtracts the values in the given histogram from this histogram."""
        for val, freq in other.items():
            self.incr(val, -freq)





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


class Pdf(object):
    """Represents a probability density function (PDF)."""

    def make_pmf(self, xs, name=""):
        """Makes a discrete version of this Pdf.

        xs: equally spaced sequence of quantities

        returns: new Pmf
        """
        ds = self.density(xs)
        pmf = Pmf(ds, xs, name=name)
        pmf.normalize()
        return pmf

    def plot(self, xs=None, **options):
        """Plots this Pdf.

        xs: sequence of quantities where the Pdf should be evaluated
        options: passed along to plt.plot
        """
        options = underride(options, label=self.name)
        if xs is None:
            low, high = self.mu - 4 * self.sigma, self.mu + 4 * self.sigma
            xs = np.linspace(low, high, 101)
        ds = self.density(xs)
        plt.plot(xs, ds, **options)


class NormalPdf(Pdf):
    """Represents the PDF of a Normal distribution."""

    def __init__(self, mu=0, sigma=1, name=""):
        """Constructs a Normal Pdf with given mu and sigma.

        mu: mean
        sigma: standard deviation
        label: string
        """
        self.mu = mu
        self.sigma = sigma
        self.name = name

    def __str__(self):
        return f"NormalPdf({self.mu}, {self.sigma}, name={self.name})"

    def density(self, xs):
        """Evaluates this Pdf at xs.

        xs: scalar or sequence of floats

        returns: float or NumPy array of probability density
        """
        return stats.norm.pdf(xs, self.mu, self.sigma)


class EstimatedPdf(Pdf):
    """Represents a PDF estimated by KDE."""

    def __init__(self, sample, name=''):
        """Estimates the density function based on a sample.

        sample: sequence of data
        label: string
        """
        self.name = name
        self.kde = stats.gaussian_kde(sample)
        self.mu = np.mean(sample)
        self.sigma = np.std(sample)

    def __str__(self):
        return f"EstimatedPdf(name={self.name})"

    def density(self, xs):
        """Evaluates this Pdf at xs.

        returns: float or NumPy array of probability density
        """
        return self.kde.evaluate(xs)

    def sample(self, n):
        """Generates a random sample from the estimated Pdf.

        n: size of sample
        """
        return self.kde.resample(n).flatten()


def credible_interval(pmf, percentage=90):
    """Computes a credible interval for a given distribution.

    If percentage=90, computes the 90% CI.

    Args:
        pmf: Pmf object representing a posterior distribution
        percentage: float between 0 and 100

    Returns:
        sequence of two floats, low and high
    """
    cdf = pmf.make_cdf()
    prob = (1 - percentage / 100) / 2
    interval = cdf.value(prob), cdf.value(1 - prob)
    return interval


def pmf_prob_less(pmf1, pmf2):
    """Probability that a value from pmf1 is less than a value from pmf2.

    Args:
        pmf1: Pmf object
        pmf2: Pmf object

    Returns:
        float probability
    """
    total = 0
    for v1, p1 in pmf1.items():
        for v2, p2 in pmf2.items():
            if v1 < v2:
                total += p1 * p2
    return total


def pmf_prob_greater(pmf1, pmf2):
    """Probability that a value from pmf1 is less than a value from pmf2.

    Args:
        pmf1: Pmf object
        pmf2: Pmf object

    Returns:
        float probability
    """
    total = 0
    for v1, p1 in pmf1.items():
        for v2, p2 in pmf2.items():
            if v1 > v2:
                total += p1 * p2
    return total


def pmf_prob_equal(pmf1, pmf2):
    """Probability that a value from pmf1 equals a value from pmf2.

    Args:
        pmf1: Pmf object
        pmf2: Pmf object

    Returns:
        float probability
    """
    total = 0
    for v1, p1 in pmf1.items():
        for v2, p2 in pmf2.items():
            if v1 == v2:
                total += p1 * p2
    return total


def random_sum(dists):
    """Chooses a random value from each dist and returns the sum.

    dists: sequence of Pmf or Cdf objects

    returns: numerical sum
    """
    total = sum(dist.random() for dist in dists)
    return total


def sample_sum(dists, n):
    """Draws a sample of sums from a list of distributions.

    dists: sequence of Pmf or Cdf objects
    n: sample size

    returns: new Pmf of sums
    """
    pmf = Pmf(random_sum(dists) for i in range(n))
    return pmf


def eval_normal_pdf(x, mu, sigma):
    """Computes the unnormalized PDF of the normal distribution.

    x: value
    mu: mean
    sigma: standard deviation

    returns: float probability density
    """
    return stats.norm.pdf(x, mu, sigma)


def make_normal_pmf(mu, sigma, num_sigmas, n=201):
    """Makes a PMF discrete approx to a Normal distribution.

    mu: float mean
    sigma: float standard deviation
    num_sigmas: how many sigmas to extend in each direction
    n: number of values in the Pmf

    returns: normalized Pmf
    """
    pmf = Pmf()
    low = mu - num_sigmas * sigma
    high = mu + num_sigmas * sigma
    for x in np.linspace(low, high, n):
        p = eval_normal_pdf(x, mu, sigma)
        pmf.set(x, p)
    pmf.normalize()
    return pmf


def eval_binomial_pmf(k, n, p):
    """Evaluates the binomial PMF.

    Returns the probabily of k successes in n trials with probability p.
    """
    return stats.binom.pmf(k, n, p)


def make_binomial_pmf(n, p):
    """Evaluates the binomial PMF.

    Returns the distribution of successes in n trials with probability p.
    """
    pmf = Pmf()
    for k in range(n + 1):
        pmf[k] = stats.binom.pmf(k, n, p)
    return pmf


def eval_gamma_pdf(x, a):
    """Computes the Gamma PDF.

    x: where to evaluate the PDF
    a: parameter of the gamma distribution

    returns: float probability
    """
    return x ** (a - 1) * np.exp(-x) / gamma(a)


def make_gamma_pmf(xs, a):
    """Makes a PMF discrete approx to a Gamma distribution.

    lam: parameter lambda in events per unit time
    xs: upper bound of the Pmf

    returns: normalized Pmf
    """
    xs = np.asarray(xs)
    ps = eval_gamma_pdf(xs, a)
    pmf = Pmf(dict(zip(xs, ps)))
    pmf.normalize()
    return pmf


def eval_geometric_pmf(k, p, loc=0):
    """Evaluates the geometric PMF.

    With loc=0: Probability of `k` trials to get one success.
    With loc=-1: Probability of `k` trials before first success.

    k: number of trials
    p: probability of success on each trial
    """
    return stats.geom.pmf(k, p, loc=loc)


def make_geometric_pmf(p, loc=0, high=10):
    """Evaluates the binomial PMF.

    With loc=0: PMF of trials to get one success.
    With loc=-1: PMF of trials before first success.

    p: probability of success
    high: upper bound where PMF is truncated
    """
    pmf = Pmf()
    for k in range(high):
        pmf[k] = stats.geom.pmf(k, p, loc=loc)
    pmf.normalize()
    return pmf


def eval_hypergeom_pmf(k, N, K, n):
    """Evaluates the hypergeometric PMF.

    Returns the probabily of k successes in n trials from a population
    N with K successes in it.
    """
    return stats.hypergeom.pmf(k, N, K, n)


def eval_poisson_pmf(k, lam):
    """Computes the Poisson PMF.

    k: number of events
    lam: parameter lambda in events per unit time

    returns: float probability
    """
    return stats.poisson.pmf(k, lam)


def make_poisson_pmf(lam, high, step=1):
    """Makes a PMF discrete approx to a Poisson distribution.

    lam: parameter lambda in events per unit time
    high: upper bound of the Pmf

    returns: normalized Pmf
    """
    pmf = Pmf()
    for k in range(0, high + 1, step):
        p = stats.poisson.pmf(k, lam)
        pmf.set(k, p)
    pmf.normalize()
    return pmf


def eval_exponential_pdf(x, lam):
    """Computes the exponential PDF.

    x: value
    lam: parameter lambda in events per unit time

    returns: float probability density
    """
    return lam * math.exp(-lam * x)


def eval_exponential_cdf(x, lam):
    """Evaluates CDF of the exponential distribution with parameter lam."""
    return 1 - math.exp(-lam * x)


def make_exponential_pmf(lam, high, n=200):
    """Makes a PMF discrete approx to an exponential distribution.

    lam: parameter lambda in events per unit time
    high: upper bound
    n: number of values in the Pmf

    returns: normalized Pmf
    """
    pmf = Pmf()
    for x in np.linspace(0, high, n):
        p = eval_exponential_pdf(x, lam)
        pmf.set(x, p)
    pmf.normalize()
    return pmf


def eval_weibull_pdf(x, lam, k):
    """Computes the Weibull PDF.

    x: value
    lam: parameter lambda in events per unit time
    k: parameter

    returns: float probability density
    """
    arg = x / lam
    return k / lam * arg ** (k - 1) * np.exp(-(arg**k))


def eval_weibull_cdf(x, lam, k):
    """Evaluates CDF of the Weibull distribution."""
    arg = x / lam
    return 1 - np.exp(-(arg**k))


def make_weibull_pmf(lam, k, high, n=200):
    """Makes a PMF discrete approx to a Weibull distribution.

    lam: parameter lambda in events per unit time
    k: parameter
    high: upper bound
    n: number of values in the Pmf

    returns: normalized Pmf
    """
    xs = np.linspace(0, high, n)
    ps = eval_weibull_pdf(xs, lam, k)
    ps[np.isinf(ps)] = 0
    return Pmf(dict(zip(xs, ps)))


def eval_pareto_pdf(x, xm, alpha):
    """Computes the Pareto.

    xm: minimum value (scale parameter)
    alpha: shape parameter

    returns: float probability density
    """
    return stats.pareto.pdf(x, alpha, scale=xm)


def make_pareto_pmf(xm, alpha, high, num=101):
    """Makes a PMF discrete approx to a Pareto distribution.

    xm: minimum value (scale parameter)
    alpha: shape parameter
    high: upper bound value
    num: number of values

    returns: normalized Pmf
    """
    xs = np.linspace(xm, high, num)
    ps = stats.pareto.pdf(xs, alpha, scale=xm)
    pmf = Pmf(dict(zip(xs, ps)))
    return pmf


def standard_normal_cdf(x):
    """Evaluates the CDF of the standard Normal distribution.

    See http://en.wikipedia.org/wiki/Normal_distribution
    #Cumulative_distribution_function

    Args:
        x: float

    Returns:
        float
    """
    root2 = math.sqrt(2)
    return (math.erf(x / root2) + 1) / 2


def eval_normal_cdf(x, mu=0, sigma=1):
    """Evaluates the CDF of the normal distribution.

    Args:
        x: float

        mu: mean parameter

        sigma: standard deviation parameter

    Returns:
        float
    """
    return stats.norm.cdf(x, loc=mu, scale=sigma)


def eval_normal_cdf_inverse(p, mu=0, sigma=1):
    """Evaluates the inverse CDF of the normal distribution.

    See http://en.wikipedia.org/wiki/Normal_distribution#Quantile_function

    Args:
        p: float

        mu: mean parameter

        sigma: standard deviation parameter

    Returns:
        float
    """
    return stats.norm.ppf(p, loc=mu, scale=sigma)


def eval_lognormal_cdf(x, mu=0, sigma=1):
    """Evaluates the CDF of the lognormal distribution.

    x: float or sequence
    mu: mean parameter
    sigma: standard deviation parameter

    Returns: float or sequence
    """
    return stats.lognorm.cdf(x, loc=mu, scale=sigma)


def render_expo_cdf(lam, low, high, n=101):
    """Generates sequences of xs and ps for an exponential CDF.

    lam: parameter
    low: float
    high: float
    n: number of points to render

    returns: numpy arrays (xs, ps)
    """
    xs = np.linspace(low, high, n)
    ps = 1 - np.exp(-lam * xs)
    return xs, ps


def render_normal_cdf(mu, sigma, low, high, n=101):
    """Generates sequences of xs and ps for a Normal CDF.

    mu: parameter
    sigma: parameter
    low: float
    high: float
    n: number of points to render

    returns: numpy arrays (xs, ps)
    """
    xs = np.linspace(low, high, n)
    ps = stats.norm.cdf(xs, mu, sigma)
    return xs, ps


def render_pareto_cdf(xmin, alpha, low, high, n=50):
    """Generates sequences of xs and ps for a Pareto CDF.

    xmin: parameter
    alpha: parameter
    low: float
    high: float
    n: number of points to render

    returns: numpy arrays (xs, ps)
    """
    if low < xmin:
        low = xmin
    xs = np.linspace(low, high, n)
    ps = 1 - (xs / xmin) ** -alpha
    return xs, ps


class Beta:
    """Represents a Beta distribution.

    See http://en.wikipedia.org/wiki/Beta_distribution
    """

    def __init__(self, alpha=1, beta=1, label=None):
        """Initializes a Beta distribution."""
        self.alpha = alpha
        self.beta = beta
        self.label = label if label is not None else "_nolegend_"

    def update(self, data):
        """Updates a Beta distribution.

        data: pair of int (heads, tails)
        """
        heads, tails = data
        self.alpha += heads
        self.beta += tails

    def mean(self):
        """Computes the mean of this distribution."""
        return self.alpha / (self.alpha + self.beta)

    def m_a_p(self):
        """Computes the value with maximum a posteori probability."""
        a = self.alpha - 1
        b = self.beta - 1
        return a / (a + b)

    def random(self):
        """Generates a random variate from this distribution."""
        return random.betavariate(self.alpha, self.beta)

    def sample(self, n):
        """Generates a random sample from this distribution.

        n: int sample size
        """
        size = (n,)
        return np.random.beta(self.alpha, self.beta, size)

    def eval_pdf(self, x):
        """Evaluates the PDF at x."""
        return x ** (self.alpha - 1) * (1 - x) ** (self.beta - 1)

    def make_pmf(self, steps=101, label=None):
        """Returns a Pmf of this distribution.

        Note: Normally, we just evaluate the PDF at a sequence
        of points and treat the probability density as a probability
        mass.

        But if alpha or beta is less than one, we have to be
        more careful because the PDF goes to infinity at x=0
        and x=1.  In that case we evaluate the CDF and compute
        differences.

        The result is a little funny, because the values at 0 and 1
        are not symmetric.  Nevertheless, it is a reasonable discrete
        model of the continuous distribution, and behaves well as
        the number of values increases.
        """
        if label is None and self.label is not None:
            label = self.label
        if self.alpha < 1 or self.beta < 1:
            cdf = self.make_cdf()
            pmf = cdf.make_pmf()
            return pmf
        xs = [(i / (steps - 1.0)) for i in range(steps)]
        probs = [self.eval_pdf(x) for x in xs]
        pmf = Pmf(dict(zip(xs, probs)), label=label)
        return pmf

    def make_cdf(self, steps=101):
        """Returns the CDF of this distribution."""
        xs = [(i / (steps - 1.0)) for i in range(steps)]
        ps = special.betainc(self.alpha, self.beta, xs)
        cdf = Cdf(xs, ps)
        return cdf

    def percentile(self, ps):
        """Returns the given percentiles from this distribution.

        ps: scalar, array, or list of [0-100]
        """
        ps = np.asarray(ps) / 100
        xs = special.betaincinv(self.alpha, self.beta, ps)
        return xs


class Dirichlet(object):
    """Represents a Dirichlet distribution.

    See http://en.wikipedia.org/wiki/Dirichlet_distribution
    """

    def __init__(self, n, conc=1, label=None):
        """Initializes a Dirichlet distribution.

        n: number of dimensions
        conc: concentration parameter (smaller yields more concentration)
        label: string label
        """
        if n < 2:
            raise ValueError("A Dirichlet dist with n<2 makes no sense")
        self.n = n
        self.params = np.ones(n, dtype=float) * conc
        self.label = label if label is not None else "_nolegend_"

    def update(self, data):
        """Updates a Dirichlet distribution.

        data: sequence of observations, in order corresponding to params
        """
        m = len(data)
        self.params[:m] += data

    def random(self):
        """Generates a random variate from this distribution.

        Returns: normalized vector of fractions
        """
        p = np.random.gamma(self.params)
        return p / p.sum()

    def likelihood(self, data):
        """Computes the likelihood of the data.

        Selects a random vector of probabilities from this distribution.

        Returns: float probability
        """
        m = len(data)
        if self.n < m:
            return 0
        x = data
        p = self.random()
        q = p[:m] ** x
        return q.prod()

    def log_likelihood(self, data):
        """Computes the log likelihood of the data.

        Selects a random vector of probabilities from this distribution.

        Returns: float log probability
        """
        m = len(data)
        if self.n < m:
            return float("-inf")
        x = self.random()
        y = np.log(x[:m]) * data
        return y.sum()

    def marginal_beta(self, i):
        """Computes the marginal distribution of the ith element.

        See http://en.wikipedia.org/wiki/Dirichlet_distribution
        #Marginal_distributions

        i: int

        Returns: Beta object
        """
        alpha0 = self.params.sum()
        alpha = self.params[i]
        return Beta(alpha, alpha0 - alpha)

    def predictive_pmf(self, xs, label=None):
        """Makes a predictive distribution.

        xs: values to go into the Pmf

        Returns: Pmf that maps from x to the mean prevalence of x
        """
        alpha0 = self.params.sum()
        ps = self.params / alpha0
        return Pmf(zip(xs, ps), label=label)


def binomial_coef(n, k):
    """Compute the binomial coefficient "n choose k".

    n: number of trials
    k: number of successes

    Returns: float
    """
    return scipy.special.comb(n, k)


def log_binomial_coef(n, k):
    """Computes the log of the binomial coefficient.

    http://math.stackexchange.com/questions/64716/
    approximating-the-logarithm-of-the-binomial-coefficient

    n: number of trials
    k: number of successes

    Returns: float
    """
    return n * math.log(n) - k * math.log(k) - (n - k) * math.log(n - k)


def normal_probability(ys, jitter=0):
    """Generates data for a normal probability plot.

    ys: sequence of values
    jitter: float magnitude of jitter added to the ys

    returns: numpy arrays xs, ys
    """
    n = len(ys)
    xs = np.random.normal(0, 1, n)
    xs.sort()
    if jitter:
        ys = jitter(ys, jitter)
    else:
        ys = np.array(ys)
    ys.sort()
    return xs, ys


def jitter(seq, std=0.5):
    """Jitters the values by adding random Gaussian noise.

    seq: sequence of numbers
    std: standard deviation of the added noise

    returns: new Numpy array
    """
    n = len(seq)
    return np.random.normal(0, std, n) + seq


def normal_probability_plot(sample, fit_color="0.8", **options):
    """Makes a normal probability plot with a fitted line.

    sample: sequence of numbers
    fit_color: color string for the fitted line
    options: passed along to Plot
    """
    xs, ys = normal_probability(sample)
    mean, var = mean_var(sample)
    std = math.sqrt(var)
    fit = fit_line(xs, mean, std)
    thinkplot.plot(*fit, color=fit_color, label="model")
    xs, ys = normal_probability(sample)
    thinkplot.plot(xs, ys, **options)


def mean(xs):
    """Computes mean.

    xs: sequence of values

    returns: float mean
    """
    return np.mean(xs)


def var(xs, mu=None, correction=0):
    """Computes variance.

    xs: sequence of values
    mu: option known mean
    ddof: delta degrees of freedom

    returns: float
    """
    xs = np.asarray(xs)
    if mu is None:
        mu = xs.mean()
    ds = xs - mu
    n = len(xs)
    return np.sum(ds**2) / (n - correction)


def std(xs, mu=None, ddof=0):
    """Computes standard deviation.

    xs: sequence of values
    mu: option known mean
    ddof: delta degrees of freedom

    returns: float
    """
    s2 = var(xs, mu, ddof)
    return math.sqrt(s2)


def mean_var(xs, ddof=0):
    """Computes mean and variance.

    Based on http://stackoverflow.com/questions/19391149/
    numpy-mean-and-variance-from-single-function

    xs: sequence of values
    ddof: delta degrees of freedom

    returns: pair of float, mean and var
    """
    xs = np.asarray(xs)
    mean = xs.mean()
    s2 = var(xs, mean, ddof)
    return mean, s2


def trim(t, p=0.01):
    """Trims the largest and smallest elements of t.

    Args:
        t: sequence of numbers
        p: fraction of values to trim off each end

    Returns:
        sequence of values
    """
    n = int(p * len(t))
    t = sorted(t)[n:-n]
    return t


def trimmed_mean(t, p=0.01):
    """Computes the trimmed mean of a sequence of numbers.

    Args:
        t: sequence of numbers
        p: fraction of values to trim off each end

    Returns:
        float
    """
    t = trim(t, p)
    return mean(t)


def trimmed_mean_var(t, p=0.01):
    """Computes the trimmed mean and variance of a sequence of numbers.

    Side effect: sorts the list.

    Args:
        t: sequence of numbers
        p: fraction of values to trim off each end

    Returns:
        float
    """
    t = trim(t, p)
    mu, var = mean_var(t)
    return mu, var


def cohen_effect_size(group1, group2):
    """Compute Cohen's d.

    group1: Series or NumPy array
    group2: Series or NumPy array

    returns: float
    """
    diff = group1.mean() - group2.mean()
    n1, n2 = len(group1), len(group2)
    var1 = group1.var()
    var2 = group2.var()
    pooled_var = (n1 * var1 + n2 * var2) / (n1 + n2)
    d = diff / math.sqrt(pooled_var)
    return d


def cov(xs, ys, xbar=None, ybar=None):
    """Computes covariance of xs and ys.

    Args:
        xs: sequence of numbers
        ys: sequence of numbers
        xbar: optional float mean of xs
        ybar: optional float mean of ys

    Returns:
        covariance of xs and ys
    """
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    if xbar is None:
        xbar = np.mean(xs)
    if ybar is None:
        ybar = np.mean(ys)
    dx = xs - xbar
    dy = ys - ybar
    cov = np.sum(dx * dy) / len(xs)
    return cov


def corr(xs, ys):
    """Computes correlation of xs and ys.

    Args:
        xs: sequence of numbers
        ys: sequence of numbers

    Returns:
        correlation of xs and ys
    """
    sx = np.std(xs)
    sy = np.std(ys)
    corr = cov(xs, ys) / sx / sy
    return corr


def serial_corr(series, lag=1):
    """Computes the serial correlation of a series.

    series: Series
    lag: integer number of intervals to shift

    returns: float correlation
    """
    xs = series[lag:]
    ys = series.shift(lag)[lag:]
    return corr(xs, ys)


def spearman_corr(xs, ys):
    """Computes Spearman's rank correlation.

    Args:
        xs: sequence of values
        ys: sequence of values

    Returns:
        float Spearman's correlation
    """
    xranks = pd.Series(xs).rank()
    yranks = pd.Series(ys).rank()
    return corr(xranks, yranks)


def map_to_ranks(t):
    """Returns a list of ranks corresponding to the elements in t.

    Args:
        t: sequence of numbers

    Returns:
        list of integer ranks, starting at 1
    """
    pairs = enumerate(t)
    sorted_pairs = sorted(pairs, key=itemgetter(1))
    ranked = enumerate(sorted_pairs)
    resorted = sorted(ranked, key=lambda trip: trip[1][0])
    ranks = [(trip[0] + 1) for trip in resorted]
    return ranks


def least_squares(xs, ys):
    """Computes a linear least squares fit for ys as a function of xs.

    Args:
        xs: sequence of numbers
        ys: sequence of numbers

    Returns:
        tuple of (intercept, slope)
    """
    xbar = mean(xs)
    ybar = mean(ys)
    slope = cov(xs, ys) / var(xs)
    inter = ybar - slope * xbar
    return inter, slope


def predict(xs, inter, slope):
    """Predicts the ys for given xs.

    xs: sequence of numbers
    inter: float intercept
    slope: float slope

    returns: NumPy array of predicted ys
    """
    xs = np.asarray(xs)
    return inter + slope * xs


def fit_line(xs, inter, slope):
    """Fits a line to the given data.

    xs: sequence of numbers
    inter: float intercept
    slope: float slope

    returns: tuple of numpy arrays (fit xs, fit ys)
    """
    low, high = min(xs), max(xs)
    fit_xs = np.linspace(low, high)
    fit_ys = predict(fit_xs, inter, slope)
    return fit_xs, fit_ys


def residuals(xs, ys, inter, slope):
    """Computes residuals for a linear fit with parameters inter and slope.

    Args:
        xs: independent variable
        ys: dependent variable
        inter: float intercept
        slope: float slope

    Returns:
        list of residuals
    """
    fit_ys = predict(xs, inter, slope)
    return ys - fit_ys


def coef_determination(ys, res):
    """Computes the coefficient of determination (R^2) for given residuals.

    Args:
        ys: dependent variable
        res: residuals

    Returns:
        float coefficient of determination
    """
    return 1 - var(res) / var(ys)


def correlated_generator(rho):
    """Generates standard normal variates with serial correlation.

    rho: target coefficient of correlation

    Returns: iterable
    """
    x = random.gauss(0, 1)
    yield x
    sigma = math.sqrt(1 - rho**2)
    while True:
        x = random.gauss(x * rho, sigma)
        yield x


def correlated_normal_generator(mu, sigma, rho):
    """Generates normal variates with serial correlation.

    mu: mean of variate
    sigma: standard deviation of variate
    rho: target coefficient of correlation

    Returns: iterable
    """
    for x in correlated_generator(rho):
        yield x * sigma + mu


def raw_moment(xs, k):
    """Computes the kth raw moment of xs."""
    return sum(x**k for x in xs) / len(xs)


def central_moment(xs, k):
    """Computes the kth central moment of xs."""
    mean = raw_moment(xs, 1)
    return sum((x - mean) ** k for x in xs) / len(xs)


def standardized_moment(xs, k):
    """Computes the kth standardized moment of xs."""
    var = central_moment(xs, 2)
    std = math.sqrt(var)
    return central_moment(xs, k) / std**k


def skewness(xs):
    """Computes skewness."""
    return standardized_moment(xs, 3)


def median(xs):
    """Computes the median (50th percentile) of a sequence.

    xs: sequence or anything else that can initialize a Cdf

    returns: float
    """
    cdf = Cdf.from_seq(xs)
    return cdf.value(0.5)


def iqr(xs):
    """Computes the interquartile of a sequence.

    xs: sequence or anything else that can initialize a Cdf

    returns: pair of floats
    """
    cdf = Cdf.from_seq(xs)
    return cdf.value(0.25), cdf.value(0.75)


def pearson_median_skewness(xs):
    """Computes the Pearson median skewness."""
    median = median(xs)
    mean = raw_moment(xs, 1)
    var = central_moment(xs, 2)
    std = math.sqrt(var)
    gp = 3 * (mean - median) / std
    return gp


class FixedWidthVariables(object):
    """Represents a set of variables in a fixed width file."""

    def __init__(self, variables, index_base=0):
        """Initializes.

        variables: DataFrame
        index_base: are the indices 0 or 1 based?

        Attributes:
        colspecs: list of (start, end) index tuples
        names: list of string variable names
        """
        self.variables = variables
        self.colspecs = variables[["start", "end"]] - index_base
        self.colspecs = self.colspecs.astype(int).values.tolist()
        self.names = variables["name"]

    def read_fixed_width(self, filename, **options):
        """Reads a fixed width ASCII file.

        filename: string filename

        returns: DataFrame
        """
        df = pd.read_fwf(filename, colspecs=self.colspecs, names=self.names, **options)
        return df


def read_stata_dct(dct_file, **options):
    """Reads a Stata dictionary file.

    dct_file: string filename
    options: dict of options passed to open()

    returns: FixedWidthVariables object
    """
    type_map = dict(
        byte=int, int=int, long=int, float=float, double=float, numeric=float
    )
    var_info = []
    with open(dct_file, **options) as f:
        for line in f:
            match = re.search("_column\\(([^)]*)\\)", line)
            if not match:
                continue
            start = int(match.group(1))
            t = line.split()
            vtype, name, fstring = t[1:4]
            name = name.lower()
            if vtype.startswith("str"):
                vtype = str
            else:
                vtype = type_map[vtype]
            long_desc = " ".join(t[4:]).strip('"')
            var_info.append((start, vtype, name, fstring, long_desc))
    columns = ["start", "type", "name", "fstring", "desc"]
    variables = pd.DataFrame(var_info, columns=columns)
    variables["end"] = variables.start.shift(-1)
    variables.loc[len(variables) - 1, "end"] = -1
    dct = FixedWidthVariables(variables, index_base=1)
    return dct


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


def percentile_row(array, p):
    """Selects the row from a sorted array that maps to percentile p.

    p: float 0--100

    returns: NumPy array (one row)
    """
    rows, cols = array.shape
    index = int(rows * p / 100)
    return array[index,]


def percentile_rows(ys_seq, percents):
    """Given a collection of lines, selects percentiles along vertical axis.

    For example, if ys_seq contains simulation results like ys as a
    function of time, and percents contains (5, 95), the result would
    be a 90% CI for each vertical slice of the simulation results.

    ys_seq: sequence of lines (y values)
    percents: list of percentiles (0-100) to select

    returns: list of NumPy arrays, one for each percentile
    """
    nrows = len(ys_seq)
    ncols = len(ys_seq[0])
    array = np.zeros((nrows, ncols))
    for i, ys in enumerate(ys_seq):
        array[i,] = ys
    array = np.sort(array, axis=0)
    rows = [percentile_row(array, p) for p in percents]
    return rows


def smooth(xs, sigma=2, **options):
    """Smooths a NumPy array with a Gaussian filter.

    xs: sequence
    sigma: standard deviation of the filter
    """
    return ndimage.filters.gaussian_filter1d(xs, sigma, **options)


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
            thinkplot.plot([x, x], [0, 1], color="0.8")

        vert_line(self.actual)
        thinkplot.cdf(self.test_cdf, label=label)

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
        hist = Hist(sample)
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
        hist = Hist(rolls)
        freqs = hist.freqs(values)
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
        hist = Hist(lengths)
        observed = np.array(hist.freqs(self.values))
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


def run_dice_test():
    """Tests whether a die is fair."""
    data = [8, 9, 19, 5, 8, 11]
    dt = DiceTest(data)
    print("dice test", dt.p_value(iters=10000))
    dt = DiceChiTest(data)
    print("dice chi test", dt.p_value(iters=10000))


def false_neg_rate(data, num_runs=1000):
    """Computes the chance of a false negative based on resampling.

    data: pair of sequences
    num_runs: how many experiments to simulate

    returns: float false negative rate
    """
    group1, group2 = data
    count = 0
    for i in range(num_runs):
        sample1 = resample(group1)
        sample2 = resample(group2)
        ht = DiffMeansPermute((sample1, sample2))
        p_value = ht.p_value(iters=101)
        if p_value > 0.05:
            count += 1
    return count / num_runs


def print_test(p_value, ht):
    """Prints results from a hypothesis test.

    p_value: float
    ht: HypothesisTest
    """
    print("p-value =", p_value)
    print("actual =", ht.actual)
    print("ts max =", ht.max_test_stat())


def quick_least_squares(xs, ys):
    """Estimates linear least squares fit and returns MSE.

    xs: sequence of values
    ys: sequence of values

    returns: inter, slope, mse
    """
    n = float(len(xs))
    xbar = xs.mean()
    dxs = xs - xbar
    varx = np.dot(dxs, dxs) / n
    ybar = ys.mean()
    dys = ys - ybar
    cov = np.dot(dxs, dys) / n
    slope = cov / varx
    inter = ybar - slope * xbar
    res = ys - (inter + slope * xs)
    mse = np.dot(res, res) / n
    return inter, slope, mse


def read_variables():
    """Reads Stata dictionary files for NSFG data.

    returns: DataFrame that maps variables names to descriptions
    """
    vars1 = read_stata_dct("2002FemPreg.dct").variables
    vars2 = read_stata_dct("2002FemResp.dct").variables
    all_vars = vars1.append(vars2)
    all_vars.index = all_vars.name
    return all_vars


def join_fem_resp(df):
    """Reads the female respondent file and joins on caseid.

    df: DataFrame
    """
    resp = nsfg.read_fem_resp()
    resp.index = resp.caseid
    join = df.join(resp, on="caseid", rsuffix="_r")
    join.screentime = pd.to_datetime(join.screentime)
    return join


MESSAGE = """If you get this error, it's probably because 
you are running Python 3 and the nice people who maintain
Patsy have not fixed this problem:
https://github.com/pydata/patsy/issues/34

While we wait, I suggest running this example in
Python 2, or skipping this example."""


def go_mining(df):
    """Searches for variables that predict birth weight.

    df: DataFrame of pregnancy records

    returns: list of (rsquared, variable name) pairs
    """
    variables = []
    for name in df.columns:
        try:
            if df[name].var() < 1e-07:
                continue
            formula = "totalwgt_lb ~ agepreg + " + name
            formula = formula.encode("ascii")
            model = smf.ols(formula, data=df)
            if model.nobs < len(df) / 2:
                continue
            results = model.fit()
        except (ValueError, TypeError):
            continue
        except patsy.PatsyError:
            raise ValueError(MESSAGE)
        variables.append((results.rsquared, name))
    return variables


def mining_report(variables, n=30):
    """Prints variables with the highest R^2.

    t: list of (R^2, variable name) pairs
    n: number of pairs to print
    """
    all_vars = read_variables()
    variables.sort(reverse=True)
    for mse, name in variables[:n]:
        key = re.sub("_r$", "", name)
        try:
            desc = all_vars.loc[key].desc
            if isinstance(desc, pd.Series):
                desc = desc[0]
            print(name, mse, desc)
        except KeyError:
            print(name, mse)


def predict_birth_weight(live):
    """Predicts birth weight of a baby at 30 weeks.

    live: DataFrame of live births
    """
    live = live[live.prglngth > 30]
    join = join_fem_resp(live)
    t = go_mining(join)
    mining_report(t)
    formula = (
        "totalwgt_lb ~ agepreg + C(race) + babysex==1 + nbrnaliv>1 + paydu==1 + totincr"
    )
    results = smf.ols(formula, data=join).fit()
    summarize_results(results)


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


def run_simple_regression(live):
    """Runs a simple regression and compare results to thinkstats2 functions.

    live: DataFrame of live births
    """
    live_dropna = live.dropna(subset=["agepreg", "totalwgt_lb"])
    ages = live_dropna.agepreg
    weights = live_dropna.totalwgt_lb
    inter, slope = least_squares(ages, weights)
    res = residuals(ages, weights, inter, slope)
    r2 = coef_determination(weights, res)
    formula = "totalwgt_lb ~ agepreg"
    model = smf.ols(formula, data=live)
    results = model.fit()
    summarize_results(results)

    def almost_equals(x, y, tol=1e-06):
        return abs(x - y) < tol

    assert almost_equals(results.params["Intercept"], inter)
    assert almost_equals(results.params["agepreg"], slope)
    assert almost_equals(results.rsquared, r2)


def pivot_tables(live):
    """Prints a pivot table comparing first babies to others.

    live: DataFrame of live births
    """
    table = pd.pivot_table(live, rows="isfirst", values=["totalwgt_lb", "agepreg"])
    print(table)


def format_row(results, columns):
    """Converts regression results to a string.

    results: RegressionResults object

    returns: string
    """
    t = []
    for col in columns:
        coef = results.params.get(col, np.nan)
        pval = results.pvalues.get(col, np.nan)
        if np.isnan(coef):
            s = "--"
        elif pval < 0.001:
            s = "%0.3g (*)" % coef
        else:
            s = "%0.3g (%0.2g)" % (coef, pval)
        t.append(s)
    try:
        t.append("%.2g" % results.rsquared)
    except AttributeError:
        t.append("%.2g" % results.prsquared)
    return t


def run_models(live):
    """Runs regressions that predict birth weight.

    live: DataFrame of pregnancy records
    """
    columns = ["isfirst[T.True]", "agepreg", "agepreg2"]
    header = ["isfirst", "agepreg", "agepreg2"]
    rows = []
    formula = "totalwgt_lb ~ isfirst"
    results = smf.ols(formula, data=live).fit()
    rows.append(format_row(results, columns))
    print(formula)
    summarize_results(results)
    formula = "totalwgt_lb ~ agepreg"
    results = smf.ols(formula, data=live).fit()
    rows.append(format_row(results, columns))
    print(formula)
    summarize_results(results)
    formula = "totalwgt_lb ~ isfirst + agepreg"
    results = smf.ols(formula, data=live).fit()
    rows.append(format_row(results, columns))
    print(formula)
    summarize_results(results)
    live["agepreg2"] = live.agepreg**2
    formula = "totalwgt_lb ~ isfirst + agepreg + agepreg2"
    results = smf.ols(formula, data=live).fit()
    rows.append(format_row(results, columns))
    print(formula)
    summarize_results(results)
    print_tabular(rows, header)


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


def logistic_regression_example():
    """Runs a simple example of logistic regression and prints results."""
    y = np.array([0, 1, 0, 1])
    x1 = np.array([0, 0, 0, 1])
    x2 = np.array([0, 1, 1, 1])
    beta = [-1.5, 2.8, 1.1]
    log_o = beta[0] + beta[1] * x1 + beta[2] * x2
    print(log_o)
    o = np.exp(log_o)
    print(o)
    p = o / (o + 1)
    print(p)
    like = y * p + (1 - y) * (1 - p)
    print(like)
    print(np.prod(like))
    df = pd.DataFrame(dict(y=y, x1=x1, x2=x2))
    results = smf.logit("y ~ x1 + x2", data=df).fit()
    print(results.summary())


def run_logistic_models(live):
    """Runs regressions that predict sex.

    live: DataFrame of pregnancy records
    """
    df = live[live.prglngth > 30]
    df["boy"] = (df.babysex == 1).astype(int)
    df["isyoung"] = (df.agepreg < 20).astype(int)
    df["isold"] = (df.agepreg < 35).astype(int)
    df["season"] = ((df.datend + 1) % 12 / 3).astype(int)
    model = smf.logit("boy ~ agepreg", data=df)
    results = model.fit()
    print("nobs", results.nobs)
    print(type(results))
    summarize_results(results)
    model = smf.logit("boy ~ agepreg + hpagelb + birthord + C(race)", data=df)
    results = model.fit()
    print("nobs", results.nobs)
    print(type(results))
    summarize_results(results)
    exog = pd.DataFrame(model.exog, columns=model.exog_names)
    endog = pd.DataFrame(model.endog, columns=[model.endog_names])
    xs = exog["agepreg"]
    lo = results.fittedvalues
    o = np.exp(lo)
    p = o / (o + 1)
    actual = endog["boy"]
    baseline = actual.mean()
    predict = results.predict() >= 0.5
    true_pos = predict * actual
    true_neg = (1 - predict) * (1 - actual)
    acc = (sum(true_pos) + sum(true_neg)) / len(actual)
    print(acc, baseline)
    columns = ["agepreg", "hpagelb", "birthord", "race"]
    new = pd.DataFrame([[35, 39, 3, 1]], columns=columns)
    y = results.predict(new)
    print(y)


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
        return math.sqrt(self.sigma2)

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
        """Cumulative probability of x.

        x: numeric
        """
        return eval_normal_cdf(x, self.mu, self.sigma)

    def percentile(self, p):
        """Inverse CDF of p.

        p: percentile rank 0-100
        """
        return eval_normal_cdf_inverse(p / 100, self.mu, self.sigma)


def normal_plot_samples(samples, plot=1, ylabel=""):
    """Makes normal probability plots for samples.

    samples: list of samples
    label: string
    """
    for n, sample in samples:
        thinkplot.sub_plot(plot)
        normal_probability_plot(sample)
        thinkplot.config(
            title="n=%d" % n, legend=False, xticks=[], yticks=[], ylabel=ylabel
        )
        plot += 1


def make_expo_samples(beta=2.0, iters=1000):
    """Generates samples from an exponential distribution.

    beta: parameter
    iters: number of samples to generate for each size

    returns: list of samples
    """
    samples = []
    for n in [1, 10, 100]:
        sample = [np.sum(np.random.exponential(beta, n)) for _ in range(iters)]
        samples.append((n, sample))
    return samples


def make_lognormal_samples(mu=1.0, sigma=1.0, iters=1000):
    """Generates samples from a lognormal distribution.

    mu: parmeter
    sigma: parameter
    iters: number of samples to generate for each size

    returns: list of samples
    """
    samples = []
    for n in [1, 10, 100]:
        sample = [np.sum(np.random.lognormal(mu, sigma, n)) for _ in range(iters)]
        samples.append((n, sample))
    return samples


def make_pareto_samples(alpha=1.0, iters=1000):
    """Generates samples from a Pareto distribution.

    alpha: parameter
    iters: number of samples to generate for each size

    returns: list of samples
    """
    samples = []
    for n in [1, 10, 100]:
        sample = [np.sum(np.random.pareto(alpha, n)) for _ in range(iters)]
        samples.append((n, sample))
    return samples


def generate_correlated(rho, n):
    """Generates a sequence of correlated values from a standard normal dist.

    rho: coefficient of correlation
    n: length of sequence

    returns: iterator
    """
    x = random.gauss(0, 1)
    yield x
    sigma = math.sqrt(1 - rho**2)
    for _ in range(n - 1):
        x = random.gauss(x * rho, sigma)
        yield x


def generate_expo_correlated(rho, n):
    """Generates a sequence of correlated values from an exponential dist.

    rho: coefficient of correlation
    n: length of sequence

    returns: NumPy array
    """
    normal = list(generate_correlated(rho, n))
    uniform = scipy.stats.norm.cdf(normal)
    expo = scipy.stats.expon.ppf(uniform)
    return expo


def make_correlated_samples(rho=0.9, iters=1000):
    """Generates samples from a correlated exponential distribution.

    rho: correlation
    iters: number of samples to generate for each size

    returns: list of samples
    """
    samples = []
    for n in [1, 10, 100]:
        sample = [np.sum(generate_expo_correlated(rho, n)) for _ in range(iters)]
        samples.append((n, sample))
    return samples


def sampling_dist_mean(data, n):
    """Computes the sampling distribution of the mean.

    data: sequence of values representing the population
    n: sample size

    returns: Normal object
    """
    mean, var = data.mean(), data.var()
    dist = Normal(mean, var)
    return dist.sum(n) / n


def plot_preg_lengths(live, firsts, others):
    """Plots sampling distribution of difference in means.

    live, firsts, others: DataFrames
    """
    print("prglngth example")
    delta = firsts.prglngth.mean() - others.prglngth.mean()
    print(delta)
    dist1 = sampling_dist_mean(live.prglngth, len(firsts))
    dist2 = sampling_dist_mean(live.prglngth, len(others))
    dist = dist1 - dist2
    print("null hypothesis", dist)
    print(dist.prob(-delta), 1 - dist.prob(delta))
    thinkplot.plot(dist, label="null hypothesis")
    thinkplot.save(root="normal3", xlabel="difference in means (weeks)", ylabel="CDF")


class CorrelationPermute(CorrelationPermute):
    """Tests correlations by permutation."""

    def test_statistic(self, data):
        """Computes the test statistic.

        data: tuple of xs and ys
        """
        xs, ys = data
        return np.corrcoef(xs, ys)[0][1]


def resample_correlations(live):
    """Tests the correlation between birth weight and mother's age.

    live: DataFrame for live births

    returns: sample size, observed correlation, CDF of resampled correlations
    """
    live2 = live.dropna(subset=["agepreg", "totalwgt_lb"])
    data = live2.agepreg.values, live2.totalwgt_lb.values
    ht = CorrelationPermute(data)
    p_value = ht.p_value()
    return len(live2), ht.actual, ht.test_cdf


def student_cdf(n):
    """Computes the CDF correlations from uncorrelated variables.

    n: sample size

    returns: Cdf
    """
    ts = np.linspace(-3, 3, 101)
    ps = scipy.stats.t.cdf(ts, df=n - 2)
    rs = ts / np.sqrt(n - 2 + ts**2)
    return Cdf(rs, ps)


def test_correlation(live):
    """Tests correlation analytically.

    live: DataFrame for live births

    """
    n, r, cdf = resample_correlations(live)
    model = student_cdf(n)
    thinkplot.plot(model.xs, model.ps, color="gray", alpha=0.3, label="Student t")
    thinkplot.cdf(cdf, label="sample")
    thinkplot.save(root="normal4", xlabel="correlation", ylabel="CDF")
    t = r * math.sqrt((n - 2) / (1 - r**2))
    p_value = 1 - scipy.stats.t.cdf(t, df=n - 2)
    print(r, p_value)


def chi_squared_cdf(n):
    """Discrete approximation of the chi-squared CDF with df=n-1.

    n: sample size

    returns: Cdf
    """
    xs = np.linspace(0, 25, 101)
    ps = scipy.stats.chi2.cdf(xs, df=n - 1)
    return Cdf(ps, xs)


def test_chi_squared():
    """Performs a chi-squared test analytically."""
    data = [8, 9, 19, 5, 8, 11]
    dt = DiceChiTest(data)
    p_value = dt.p_value(iters=1000)
    n, chi2, cdf = len(data), dt.actual, dt.test_cdf
    model = chi_squared_cdf(n)
    thinkplot.plot(model.xs, model.ps, color="gray", alpha=0.3, label="chi squared")
    thinkplot.Cdf(cdf, label="sample")
    thinkplot.save(
        root="normal5", xlabel="chi-squared statistic", ylabel="CDF", loc="lower right"
    )
    p_value = 1 - scipy.stats.chi2.cdf(chi2, df=n - 1)
    print(chi2, p_value)


def make_clt_plots():
    """Makes plot showing distributions of sums converging to normal."""
    thinkplot.pre_plot(num=3, rows=2, cols=3)
    samples = make_expo_samples()
    normal_plot_samples(samples, plot=1, ylabel="sum of expo values")
    thinkplot.pre_plot(num=3)
    samples = make_lognormal_samples()
    normal_plot_samples(samples, plot=4, ylabel="sum of lognormal values")
    thinkplot.save(root="normal1", legend=False)
    thinkplot.pre_plot(num=3, rows=2, cols=3)
    samples = make_pareto_samples()
    normal_plot_samples(samples, plot=1, ylabel="sum of Pareto values")
    thinkplot.pre_plot(num=3)
    samples = make_correlated_samples()
    normal_plot_samples(samples, plot=4, ylabel="sum of correlated expo values")
    thinkplot.save(root="normal2", legend=False)


class SurvivalFunction(object):
    """Represents a survival function."""

    def __init__(self, ts, ss, label=""):
        self.ts = ts
        self.ss = ss
        self.label = label

    def __len__(self):
        return len(self.ts)

    def __getitem__(self, t):
        return self.prob(t)

    def prob(self, t):
        """Returns S(t), the probability that corresponds to value t.
        t: time
        returns: float probability
        """
        return np.interp(t, self.ts, self.ss, left=1.0)

    def probs(self, ts):
        """Gets probabilities for a sequence of values."""
        return np.interp(ts, self.ts, self.ss, left=1.0)

    def items(self):
        """Sorted sequence of (t, s) pairs."""
        return zip(self.ts, self.ss)

    def render(self):
        """Generates a sequence of points suitable for plotting.
        returns: tuple of (sorted times, survival function)
        """
        return self.ts, self.ss

    def make_hazard_function(self, label=""):
        """Computes the hazard function.

        This simple version does not take into account the
        spacing between the ts.  If the ts are not equally
        spaced, it is not valid to compare the magnitude of
        the hazard function across different time steps.

        label: string

        returns: HazardFunction object
        """
        lams = pd.Series(index=self.ts)
        prev = 1.0
        for t, s in zip(self.ts, self.ss):
            lams[t] = (prev - s) / prev
            prev = s
        return HazardFunction(lams, label=label)

    def make_pmf(self, filler=None):
        """Makes a PMF of lifetimes.

        filler: value to replace missing values

        returns: Pmf
        """
        cdf = Cdf(self.ts, 1 - self.ss)
        pmf = Pmf()
        for val, prob in cdf.items():
            pmf.set(val, prob)
        cutoff = cdf.ps[-1]
        if filler is not None:
            pmf[filler] = 1 - cutoff
        return pmf

    def remaining_lifetime(self, filler=None, func=Pmf.mean):
        """Computes remaining lifetime as a function of age.
        func: function from conditional Pmf to expected liftime
        returns: Series that maps from age to remaining lifetime
        """
        pmf = self.make_pmf(filler=filler)
        d = {}
        for t in sorted(pmf.values())[:-1]:
            pmf[t] = 0
            pmf.normalize()
            d[t] = func(pmf) - t
        return pd.Series(d)


def make_survival_from_seq(values, label=""):
    """Makes a survival function based on a complete dataset.

    values: sequence of observed lifespans

    returns: SurvivalFunction
    """
    counter = Counter(values)
    ts, freqs = zip(*sorted(counter.items()))
    ts = np.asarray(ts)
    ps = np.cumsum(freqs, dtype=np.float)
    ps /= ps[-1]
    ss = 1 - ps
    return SurvivalFunction(ts, ss, label)


def make_survival_from_cdf(cdf, label=""):
    """Makes a survival function based on a CDF.

    cdf: Cdf

    returns: SurvivalFunction
    """
    ts = cdf.xs
    ss = 1 - cdf.ps
    return SurvivalFunction(ts, ss, label)


class HazardFunction(object):
    """Represents a hazard function."""

    def __init__(self, d, label=""):
        """Initialize the hazard function.

        d: dictionary (or anything that can initialize a series)
        label: string
        """
        self.series = pd.Series(d)
        self.label = label

    def __len__(self):
        return len(self.series)

    def __getitem__(self, t):
        return self.series[t]

    def get(self, t, default=np.nan):
        return self.series.get(t, default)

    def render(self):
        """Generates a sequence of points suitable for plotting.

        returns: tuple of (sorted times, hazard function)
        """
        return self.series.index, self.series.values

    def make_survival(self, label=""):
        """Makes the survival function.

        returns: SurvivalFunction
        """
        ts = self.series.index
        ss = (1 - self.series).cumprod()
        sf = SurvivalFunction(ts, ss, label=label)
        return sf

    def extend(self, other):
        """Extends this hazard function by copying the tail from another.
        other: HazardFunction
        """
        last_index = self.series.index[-1] if len(self) else 0
        more = other.series[other.series.index > last_index]
        self.series = pd.concat([self.series, more])

    def truncate(self, t):
        """Truncates this hazard function at the given value of t.
        t: number
        """
        self.series = self.series[self.series.index < t]


def conditional_survival(pmf, t0):
    """Computes conditional survival function.

    Probability that duration exceeds t0+t, given that
    duration >= t0.

    pmf: Pmf of durations
    t0: minimum time

    returns: tuple of (ts, conditional survivals)
    """
    cond = Pmf()
    for t, p in pmf.items():
        if t >= t0:
            cond.set(t - t0, p)
    cond.normalize()
    return make_survival_from_cdf(cond.make_cdf())


def plot_conditional_survival(durations):
    """Plots conditional survival curves for a range of t0.

    durations: list of durations
    """
    pmf = Pmf(durations)
    times = [8, 16, 24, 32]
    thinkplot.pre_plot(len(times))
    for t0 in times:
        sf = conditional_survival(pmf, t0)
        label = "t0=%d" % t0
        thinkplot.plot(sf, label=label)
    thinkplot.show()


def plot_survival(complete):
    """Plots survival and hazard curves.

    complete: list of complete lifetimes
    """
    thinkplot.pre_plot(3, rows=2)
    cdf = Cdf.from_seq(complete, label="cdf")
    sf = make_survival_from_cdf(cdf, label="survival")
    print(cdf[13])
    print(sf[13])
    thinkplot.plot(sf)
    thinkplot.cdf(cdf, alpha=0.2)
    thinkplot.config()
    thinkplot.sub_plot(2)
    hf = sf.make_hazard_function(label="hazard")
    print(hf[39])
    thinkplot.plot(hf)
    thinkplot.config(ylim=[0, 0.75])


def plot_hazard(complete, ongoing):
    """Plots the hazard function and survival function.

    complete: list of complete lifetimes
    ongoing: list of ongoing lifetimes
    """
    sf = make_survival_from_seq(complete)
    thinkplot.plot(sf, label="old S(t)", alpha=0.1)
    thinkplot.pre_plot(2)
    hf = estimate_hazard_function(complete, ongoing)
    thinkplot.plot(hf, label="lams(t)", alpha=0.5)
    sf = hf.make_survival()
    thinkplot.plot(sf, label="S(t)")
    thinkplot.show(xlabel="t (weeks)")


def estimate_hazard_function(complete, ongoing, label="", verbose=False):
    """Estimates the hazard function by Kaplan-Meier.

    http://en.wikipedia.org/wiki/Kaplan%E2%80%93Meier_estimator

    complete: list of complete lifetimes
    ongoing: list of ongoing lifetimes
    label: string
    verbose: whether to display intermediate results
    """
    if np.sum(np.isnan(complete)):
        raise ValueError("complete contains NaNs")
    if np.sum(np.isnan(ongoing)):
        raise ValueError("ongoing contains NaNs")
    hist_complete = Counter(complete)
    hist_ongoing = Counter(ongoing)
    ts = list(hist_complete | hist_ongoing)
    ts.sort()
    at_risk = len(complete) + len(ongoing)
    lams = pd.Series(index=ts)
    for t in ts:
        ended = hist_complete[t]
        censored = hist_ongoing[t]
        lams[t] = ended / at_risk
        if verbose:
            print("%0.3g\t%d\t%d\t%d\t%0.2g" % (t, at_risk, ended, censored, lams[t]))
        at_risk -= ended + censored
    return HazardFunction(lams, label=label)


def estimate_hazard_numpy(complete, ongoing, label=""):
    """Estimates the hazard function by Kaplan-Meier.

    Just for fun, this is a version that uses NumPy to
    eliminate loops.

    complete: list of complete lifetimes
    ongoing: list of ongoing lifetimes
    label: string
    """
    hist_complete = Counter(complete)
    hist_ongoing = Counter(ongoing)
    ts = set(hist_complete) | set(hist_ongoing)
    at_risk = len(complete) + len(ongoing)
    ended = [hist_complete[t] for t in ts]
    ended_c = np.cumsum(ended)
    censored_c = np.cumsum([hist_ongoing[t] for t in ts])
    not_at_risk = np.roll(ended_c, 1) + np.roll(censored_c, 1)
    not_at_risk[0] = 0
    at_risk_array = at_risk - not_at_risk
    hs = ended / at_risk_array
    lams = dict(zip(ts, hs))
    return HazardFunction(lams, label=label)


def add_labels_by_decade(groups, **options):
    """Draws fake points in order to add labels to the legend.

    groups: GroupBy object
    """
    thinkplot.pre_plot(len(groups))
    for name, _ in groups:
        label = "%d0s" % name
        thinkplot.plot([15], [1], label=label, **options)


def estimate_marriage_survival_by_decade(groups, **options):
    """Groups respondents by decade and plots survival curves.

    groups: GroupBy object
    """
    thinkplot.pre_plot(len(groups))
    for _, group in groups:
        _, sf = estimate_marriage_survival(group)
        thinkplot.plot(sf, **options)


def plot_predictions_by_decade(groups, **options):
    """Groups respondents by decade and plots survival curves.

    groups: GroupBy object
    """
    hfs = []
    for _, group in groups:
        hf, sf = estimate_marriage_survival(group)
        hfs.append(hf)
    thinkplot.pre_plot(len(hfs))
    for i, hf in enumerate(hfs):
        if i > 0:
            hf.extend(hfs[i - 1])
        sf = hf.make_survival()
        thinkplot.plot(sf, **options)


def resample_survival(resp, iters=101):
    """Resamples respondents and estimates the survival function.

    resp: DataFrame of respondents
    iters: number of resamples
    """
    _, sf = estimate_marriage_survival(resp)
    thinkplot.plot(sf)
    low, high = resp.agemarry.min(), resp.agemarry.max()
    ts = np.arange(low, high, 1 / 12.0)
    ss_seq = []
    for _ in range(iters):
        sample = resample_rows_weighted(resp)
        _, sf = estimate_marriage_survival(sample)
        ss_seq.append(sf.probs(ts))
    low, high = percentile_rows(ss_seq, [5, 95])
    thinkplot.fill_between(ts, low, high, color="gray", label="90% CI")
    thinkplot.save(
        root="survival3",
        xlabel="age (years)",
        ylabel="prob unmarried",
        xlim=[12, 46],
        ylim=[0, 1],
    )


def estimate_marriage_survival(resp):
    """Estimates the survival curve.

    resp: DataFrame of respondents

    returns: pair of HazardFunction, SurvivalFunction
    """
    complete = resp[resp.evrmarry == 1].agemarry.dropna()
    ongoing = resp[resp.evrmarry == 0].age
    hf = estimate_hazard_function(complete, ongoing)
    sf = hf.make_survival()
    return hf, sf


def plot_marriage_data(resp):
    """Plots hazard and survival functions.

    resp: DataFrame of respondents
    """
    hf, sf = estimate_marriage_survival(resp)
    thinkplot.pre_plot(rows=2)
    thinkplot.plot(hf)
    thinkplot.config(ylabel="hazard", legend=False)
    thinkplot.sub_plot(2)
    thinkplot.plot(sf)
    thinkplot.save(
        root="survival2",
        xlabel="age (years)",
        ylabel="prob unmarried",
        ylim=[0, 1],
        legend=False,
    )
    return sf


def plot_pregnancy_data(preg):
    """Plots survival and hazard curves based on pregnancy lengths.

    preg:


    Outcome codes from http://www.icpsr.umich.edu/nsfg6/Controller?
    displayPage=labelDetails&fileCode=PREG&section=&subSec=8016&srtLabel=611932

    1	LIVE BIRTH	 	9148
    2	INDUCED ABORTION	1862
    3	STILLBIRTH	 	120
    4	MISCARRIAGE	 	1921
    5	ECTOPIC PREGNANCY	190
    6	CURRENT PREGNANCY	352

    """
    complete = preg.query("outcome in [1, 3, 4]").prglngth
    print("Number of complete pregnancies", len(complete))
    ongoing = preg[preg.outcome == 6].prglngth
    print("Number of ongoing pregnancies", len(ongoing))
    plot_survival(complete)
    thinkplot.save(root="survival1", xlabel="t (weeks)")
    hf = estimate_hazard_function(complete, ongoing)
    sf = hf.make_survival()
    return sf


def plot_remaining_lifetime(sf1, sf2):
    """Plots remaining lifetimes for pregnancy and age at first marriage.

    sf1: SurvivalFunction for pregnancy length
    sf2: SurvivalFunction for age at first marriage
    """
    thinkplot.pre_plot(cols=2)
    rem_life1 = sf1.remaining_lifetime()
    thinkplot.plot(rem_life1)
    thinkplot.config(
        title="remaining pregnancy length",
        xlabel="weeks",
        ylabel="mean remaining weeks",
    )
    thinkplot.sub_plot(2)
    func = lambda pmf: pmf.percentile(50)
    rem_life2 = sf2.remaining_lifetime(filler=np.inf, func=func)
    thinkplot.plot(rem_life2)
    thinkplot.config(
        title="years until first marriage",
        ylim=[0, 15],
        xlim=[11, 31],
        xlabel="age (years)",
        ylabel="median remaining years",
    )
    thinkplot.save(root="survival6")


def plot_resampled_by_decade(resps, iters=11, predict_flag=False, omit=None):
    """Plots survival curves for resampled data.

    resps: list of DataFrames
    iters: number of resamples to plot
    predict_flag: whether to also plot predictions
    """
    for i in range(iters):
        samples = [resample_rows_weighted(resp) for resp in resps]
        sample = pd.concat(samples, ignore_index=True)
        groups = sample.groupby("decade")
        if omit:
            groups = [(name, group) for name, group in groups if name not in omit]
        if i == 0:
            add_labels_by_decade(groups, alpha=0.7)
        if predict_flag:
            plot_predictions_by_decade(groups, alpha=0.1)
            estimate_marriage_survival_by_decade(groups, alpha=0.1)
        else:
            estimate_marriage_survival_by_decade(groups, alpha=0.2)


def read_baby_boom(filename="babyboom.dat"):
    """Reads the babyboom data.

    filename: string

    returns: DataFrame
    """
    var_info = [
        ("time", 1, 8, int),
        ("sex", 9, 16, int),
        ("weight_g", 17, 24, int),
        ("minutes", 25, 32, int),
    ]
    columns = ["name", "start", "end", "type"]
    variables = pd.DataFrame(var_info, columns=columns)
    variables.end += 1
    dct = FixedWidthVariables(variables, index_base=1)
    df = dct.read_fixed_width(filename, skiprows=59)
    return df

##  Plotting functions

def pmf_bar_plots(pmf1, pmf2, width=0.45, **options):
    """
    """
    pmf1.bar(align="edge", width=-width, **options)
    pmf2.bar(align="edge", width=width, **options)


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
    loc = options.pop('loc', 'best')
    if options.pop('legend', True):
        legend(loc=loc)

    plt.gca().set(**options)
    plt.tight_layout()


def legend(**options):
    """Draws a legend only if there is at least one labeled item.
    options are passed to plt.legend()
    https://matplotlib.org/api/_as_gen/matplotlib.pyplot.legend.html
    """
    underride(options, loc='best')

    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, **options)



def main():
    pass


if __name__ == "__main__":
    main()
