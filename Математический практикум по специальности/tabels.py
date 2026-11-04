import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import norm, binom, poisson, bernoulli, uniform, expon, cauchy, geom, gamma
from statsmodels.distributions.empirical_distribution import ECDF

### вспомогательные функции
x = np.linspace(-4, 4, 1000)

### функции плотности
norm.pdf(x, loc=0, scale=1)              # нормальное распределение
binom.pmf(k=50, n=100, p=0.5)            # биномиальное распределение
poisson.pmf(k=5, mu=3)                   # распределение Пуассона
bernoulli.pmf(k=1, p=0.5)                # распределение Бернулли
uniform.pdf(x=0.5, loc=0, scale=1)       # равномерное распределение
expon.pdf(x=1, scale=1)                  # экспоненциальное распределение
cauchy.pdf(x=0, loc=0, scale=1)          # распределение Коши
geom.pmf(k=3, p=0.5)                     # геометрическое распределение
gamma.pdf(x=2, a=2, scale=1)             # гамма-распределение

### функции распределения (CDF)
norm.cdf(x, loc=0, scale=1)              # нормальное распределение
binom.cdf(k=50, n=100, p=0.5)            # биномиальное распределение
poisson.cdf(k=5, mu=3)                   # распределение Пуассона
bernoulli.cdf(k=1, p=0.5)                # распределение Бернулли
uniform.cdf(x=0.5, loc=0, scale=1)       # равномерное распределение
expon.cdf(x=1, scale=1)                  # экспоненциальное распределение
cauchy.cdf(x=0, loc=0, scale=1)          # распределение Коши
geom.cdf(k=3, p=0.5)                     # геометрическое распределение
gamma.cdf(x=2, a=2, scale=1)             # гамма-распределение

### функции обратного распределения (PPF/quantiles)
norm.ppf(q=0.975, loc=0, scale=1)        # нормальное распределение
binom.ppf(q=0.95, n=100, p=0.5)          # биномиальное распределение
poisson.ppf(q=0.95, mu=3)                # распределение Пуассона
bernoulli.ppf(q=0.7, p=0.5)              # распределение Бернулли
uniform.ppf(q=0.75, loc=0, scale=1)      # равномерное распределение
expon.ppf(q=0.95, scale=1)               # экспоненциальное распределение
cauchy.ppf(q=0.9, loc=0, scale=1)        # распределение Коши
geom.ppf(q=0.8, p=0.5)                   # геометрическое распределение
gamma.ppf(q=0.9, a=2, scale=1)           # гамма-распределение

### генерация случайных величин
binom.rvs(n=100, p=0.04, size=1000)      # биномиальное распределение
poisson.rvs(mu=1, size=1000)             # распределение Пуассона
bernoulli.rvs(0.5, size=1000)            # распределение Бернулли
uniform.rvs(0, 1, size=1000)             # равномерное распределение
expon.rvs(scale=1, size=1000)            # экспоненциальное распределение
cauchy.rvs(0, 1, size=1000)              # распределение Коши
geom.rvs(p=0.5, size=1000)               # геометрическое распределение (исправлено: p вместо 1)
gamma.rvs(a=2, scale=1, size=1000)       # гамма-распределение (исправлено: a вместо loc)
norm.rvs(loc=0, scale=1, size=1000)      # нормальное распределение (добавлено)

### дополнительные функции numpy
np.random.uniform(0, 1, 1000)            # равномерное распределение
np.random.exponential(1, 1000)           # экспоненциальное распределение
np.random.standard_cauchy(1000)          # стандартное распределение Коши
np.random.normal(0, 1, 1000)             # нормальное распределение (добавлено)
np.random.binomial(n=100, p=0.5, size=1000)  # биномиальное распределение (добавлено)
np.random.poisson(lam=3, size=1000)      # распределение Пуассона (добавлено)
np.random.geometric(p=0.5, size=1000)    # геометрическое распределение (добавлено)