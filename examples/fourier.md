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

# Fourier Components Don't Phase Me

This notebook contains a supporting example for [Thinks Stats 3e](https://allendowney.github.io/ThinkStats/).

The third edition is available now from [Bookshop.org](https://bookshop.org/a/98697/9781098190255) and [Amazon](https://amzn.to/42lmxwu) (those are affiliate links). If you find this book useful, please consider [supporting free, open educational materials](https://greenteapress.com/support/?book=ts3) from Green Tea Press.

[Click here to run this notebook on Colab](https://colab.research.google.com/github/AllenDowney/ThinkStats/blob/v3/examples/fourier.ipynb)

```python tags=["hide-cell"]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.printoptions(legacy='1.25')
plt.rcParams["figure.dpi"] = 100
plt.rcParams["figure.figsize"] = [6, 3.5]
```

Coincidentally, the use of Fourier components in regression models has come up twice this week, on two separate projects.
This notebook presents a small example to show how they work.

Note: The title of this article is a play on the [homophones "phase" and "faze"](https://www.merriam-webster.com/grammar/phase-vs-faze).

Suppose we have a signal that contains a periodic component parameterized by frequency `f`, amplitude `A`, and phase shift `phi`.
I'll give these parameters arbitrary values and we'll see if we can recover them.

```python
f = 1
A = 1.5
phi = 0.5
```

To synthesize the signal, I'll evaluate a cosine with these parameters over three cycles.

```python
ts = np.linspace(0, 3, endpoint=False)
ys = A * np.cos(2 * np.pi * f * ts + phi)
np.mean(ys)
```

And add in zero-mean Gaussian noise.

```python
np.random.seed(1)
noise = np.random.normal(0, 0.5, size=len(ts))
ys += noise - noise.mean()
```

Here's what the signal looks like.

```python
plt.plot(ts, ys)
plt.xlabel('t')
plt.ylabel('y');
```

Now let's see if we can use the signal to estimate the parameters.
The key idea is this trigonometric identity:

$$ \cos(\omega t + \phi) = \cos\phi \cos(\omega t) - \sin\phi \sin(\omega t) $$

In words, a cosine with a phase shift can be expressed as the weighted sum of a sine and cosine with no phase shift.
So you can estimate the amplitude and phase of a periodic signal by running a regression with a sine and cosine as predictors -- provided that you know the frequency of the periodic component.

To demonstrate, I'll put the data in a `DataFrame` along with these Fourier components.

```python
data = pd.DataFrame(dict(ys=ys, ts=ts))
data['cos'] = np.cos(2 * np.pi * f * ts)
data['sin'] = np.sin(2 * np.pi * f * ts)
```

Now we can run the regression model (excluding the intercept, since we didn't include one in the synthesized signal).

```python
import statsmodels.formula.api as smf

res = smf.ols('ys ~ 0 + cos + sin', data=data).fit()
res.summary()
```

Here's what the fitted model looks like compared to the data.

```python
data['yhat'] = res.fittedvalues

plt.plot(data['ts'], data['ys'], label='data', alpha=0.6)
plt.plot(data['ts'], data['yhat'], label='fit')
plt.xlabel('t')
plt.ylabel('y')
plt.legend();
```

It looks like the fitted curve has recovered the phase of the periodic component.
The estimated amplitude and phase are not represented explicitly in the parameters of the model, but we can compute them -- basically by converting them from Cartesian to polar coordinates.

```python
a = res.params['cos']
b = res.params['sin']

A_hat = np.hypot(a, b)     
phi_hat = np.arctan2(-b, a)

A_hat, phi_hat
```

It looks like we recovered the parameters, at least approximately.

Fourier analysis works on pretty much the same principle.
Instead of using sine and cosine, it uses a complex exponential.

```python
ws = np.exp(2 * np.pi * 1j * f * ts)
```

We can estimate the parameters by computing the dot product of the complex exponential with the signal (which is basically correlation).

```python
c = 2 / len(ts) * np.vdot(ws, ys)
c
```

The result is a complex number that represents the estimated amplitude and phase of the signal.

```python
A_hat = np.abs(c)
phi_hat = np.angle(c)

A_hat, phi_hat
```

And the estimated values are the same as what we got from linear regression.

This works because $ e^{i\theta} = \cos\theta + i\sin\theta $, so when we project the signal on a complex exponential basis, we're effectively computing the sine and cosine components in a compact form.


```python

```

[Think Stats: Exploratory Data Analysis in Python, 3rd Edition](https://allendowney.github.io/ThinkStats/index.html)

Copyright 2024 [Allen B. Downey](https://allendowney.com)

Code license: [MIT License](https://mit-license.org/)

Text license: [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/)
