---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Rip-off ETF?

This notebook contains a supporting example for [Thinks Stats 3e](https://allendowney.github.io/ThinkStats/).

The third edition is available now from [Bookshop.org](https://bookshop.org/a/98697/9781098190255) and [Amazon](https://amzn.to/42lmxwu) (those are affiliate links). If you find this book useful, please consider [supporting free, open educational materials](https://greenteapress.com/support/?book=ts3) from Green Tea Press.

[Click here to run this notebook on Colab](https://colab.research.google.com/github/AllenDowney/ThinkStats/blob/v3/examples/ripoff_etf.ipynb)

```python tags=["hide-cell"]
import os
import urllib.request
import urllib.parse

def download(url):
    filename = os.path.basename(urllib.parse.unquote(url))
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url, filename)
        print("Downloaded " + filename)


download("https://github.com/AllenDowney/ThinkStats/raw/v3/nb/thinkstats.py")
```

```python tags=["hide-cell"]
try:
    import empiricaldist
except ImportError:
    !pip install empiricaldist
```

```python tags=["hide-cell"]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from thinkstats import decorate
```

An article in a recent issue of  *The Economist* suggests, right in the title, "[Investors should avoid a new generation of rip-off ETFs](https://www.economist.com/finance-and-economics/2024/08/22/investors-should-avoid-a-new-generation-of-rip-off-etfs)".
An ETF is an exchange-traded fund, which holds a collection of assets and trades on an exchange like a single stock. For example, the SPDR S&P 500 ETF Trust (SPY) tracks the S&P 500 index, but unlike traditional index funds, you can buy or sell shares in minutes.

There's nothing obviously wrong with that -- but as an example of a "rip-off ETF", the article describes "defined-outcome funds" or buffer ETFs, which "offer investors an enviable-sounding opportunity: hold stocks, with protection against falling prices. All they must do is forgo annual returns above a certain level, often 10% or so."

That might sound good, but the article explains, "Over the long term, they are a terrible deal for investors. Much of the compounding effect of stock ownership comes from rallies."

To demonstrate, they use the value of the S&P index since 1980: "An investor with returns capped at 10% and protected from losses would have made a real return of 403% over the period, a fraction of the 3,155% return offered by just buying and holding the S&P 500."

So that sounds bad, but returns from 1980 to the present have been historically unusual.
To get a sense of whether buffer ETFs are more generally a bad deal, let's get a bigger picture.

<!-- #region tags=["hide-cell"] -->
## The Dow Jones

The [MeasuringWorth Foundation](https://www.measuringworth.com/datasets/DJA) has compiled the value of the Dow Jones Industrial Average at the end of each day from February 16, 1885 to the present, with adjustments at several points to make the values comparable.
The series I collected starts on February 16, 1885 and ends on August 30, 2024.
The following cells download and read the data.
<!-- #endregion -->

```python tags=["hide-cell"]
# "Citation: Samuel H. Williamson, 'Daily Closing Value of the Dow Jones Average, 1885 to Present,'
# MeasuringWorth, 2022. "

# Downloaded from https://www.measuringworth.com/datasets/DJA, September 3, 2024
```

```python
DATA_PATH = "https://github.com/AllenDowney/ThinkStats/raw/v3/data/"
filename = "DJA.csv"
download(DATA_PATH + filename)
```

```python
djia = pd.read_csv(filename, skiprows=4, parse_dates=[0], index_col=0)
djia.head()
```

To compute annual returns, we'll start by selecting the closing price on the last trading day of each year (dropping 2024 because we don't have a complete year).

```python
annual = djia.groupby(djia.index.year).last().drop(2024)
annual
```

Next we'll compute the annual price return, which is the ratio of successive year-end closing prices.

```python
annual['Ratio'] = annual['DJIA'] / annual['DJIA'].shift(1)
annual
```

And the relative return as a percentage.

```python
annual['Return'] = (annual['Ratio'] - 1) * 100
```

Looking at the years with the biggest losses and gains, we can see that most of the extremes were before the 1960s -- with the exception of the 2008 financial crisis.

```python
annual.dropna().sort_values(by='Return')
```

Here's what the distribution of annual returns looks like.

```python
from empiricaldist import Cdf

cdf_return = Cdf.from_seq(annual['Return'])
cdf_return.plot()

decorate(xlabel='Annual return (percent)', ylabel='CDF')
plt.savefig('ripoff_etf1.png', dpi=300)
```

Immediately we see why capping returns at 10% might be a bad idea -- this cap is exceeded almost 45% of the time, and sometimes by a lot!

```python
1 - cdf_return(10)
```

## Long-Term Returns

We'll use the following function to compute long-term returns.
It takes a start date and a duration, and computes two ratios:

* The total price return based on actual annual returns.

* The total price return if annual returns are clipped at 0 and 10 -- that is, any negative returns are set to 0 and any returns above 10 are set to 10.

```python
def compute_ratios(start=1993, duration=30):
    end = start + duration
    interval = annual.loc[start: end]
    ratio = interval['Ratio'].prod()
    low, high = 1.0, 1.10
    clipped = interval['Ratio'].clip(low, high)
    ratio_clipped = clipped.prod()
    return start, end, ratio, ratio_clipped
```

With this function, we can replicate the analysis *The Economist* did with the S&P 500.
Here are the results for the DJIA from the beginning of 1980 to the end of 2023.

```python
compute_ratios(1980, 43)
```

A buffer ETF over this period would have grown by a factor of more than 15 in nominal dollars, with no risk of loss.
But an index fund would have grown by a factor of almost 45.
So yeah, the ETF would have been a bad deal.

However, if we go back to the bad old days, an investor in 1900 would have been substantially better off with a buffer ETF held for 43 years -- a factor of 7.2 compared to a factor of 2.8.

```python
compute_ratios(1900, 43)
```

It seems we can cherry-pick the data to make the comparison go either way -- so let's see how things look more generally.
Starting in 1886, we'll compute price returns for all 30-year intervals, ending with the interval from 1993 to 2023.

```python
duration = 30
ratios = [compute_ratios(start, duration) for start in range(1886, 2024-duration)]
ratios = pd.DataFrame(ratios, columns=['Start', 'End', 'Index Fund', 'Buffer ETF'])
ratios.index = ratios['Start']
ratios.tail()
```

Here's what the returns look like for an index fund compared to a buffer ETF.

```python
ratios['Index Fund'].plot()
ratios['Buffer ETF'].plot()

decorate(xlabel='Start year', ylabel='30-year price return')
plt.savefig('ripoff_etf2.png', dpi=300)
```

The buffer ETF performs as advertised, substantially reducing volatility.
But it has only occasionally been a good deal, and not in my lifetime.

According to ChatGPT, the primary reasons for strong growth in stock prices since the 1960s are "technological advancements, globalization, financial market innovation, and favorable monetary policies".
If you think these elements will generally persist over the next 30 years, you might want to avoid buffer ETFs. 

```python

```

[Think Stats: Exploratory Data Analysis in Python, 3rd Edition](https://allendowney.github.io/ThinkStats/index.html)

Copyright 2024 [Allen B. Downey](https://allendowney.com)

Code license: [MIT License](https://mit-license.org/)

Text license: [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/)
