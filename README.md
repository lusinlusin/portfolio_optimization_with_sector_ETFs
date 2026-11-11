English | [简体中文](README.zh-CN.md)

# When Mean-Variance Optimization Meets Real Data

What happens to the textbook portfolio model when its inputs have to be estimated, and which
of the standard remedies deal with the cause rather than the symptom? A rolling
out-of-sample study on the eleven SPDR S&P 500 sector ETFs, 2019–2025.

## The model: mean-variance optimization

Mean-variance optimization (MVO) chooses portfolio weights `x`, one number per asset, from
two inputs:

- `μ`: the expected return of each asset
- `Q`: the covariance matrix of the returns

It solves

```math
\max_{x} \;\; \mu^\top x \;-\; \lambda\, x^\top Q\, x \qquad \text{subject to} \qquad e^\top x = 1
```

| Term | Meaning |
|---|---|
| `μᵀx` | expected return of the portfolio |
| `xᵀQx` | variance of the portfolio |
| `λ` | risk aversion: how much one unit of variance costs in units of return |
| `eᵀx = 1` | budget: the weights sum to 1 (`e` is a vector of ones); short selling is allowed |

The problem has a closed-form solution:

```math
x^\star = \frac{1}{2\lambda}\, Q^{-1}\left(\mu - \gamma e\right),
\qquad
\gamma = \frac{e^\top Q^{-1}\mu - 2\lambda}{e^\top Q^{-1} e}
```

The part to notice is `Q⁻¹`. **The solution exists only if `Q` can be inverted.**

## Challenge 1: nobody knows `μ` and `Q`

The model takes `μ` and `Q` as given. In reality neither can be observed. Both have to be
estimated from the last `L` months of returns `r₁, …, r_L`:

```math
\hat{\mu} = \frac{1}{L}\sum_{t=1}^{L} r_t,
\qquad
\hat{Q} = \frac{1}{L-1}\sum_{t=1}^{L} \left(r_t - \hat{\mu}\right)\left(r_t - \hat{\mu}\right)^\top
```

MVO then uses `μ̂` and `Q̂` as if they were the true values.

- With a very long history, the estimates would be close to the truth and this would be harmless.
- In practice the usable history is short, because markets change and old data stops being relevant.
- So the optimizer works from noisy inputs and has no way of knowing that they are noisy.

## Challenge 2: with little data, the problem cannot be solved

Take the setting of this study, `n = 11` assets, and suppose there are only `L = 6` months
of returns.

1. **`Q̂` does not have full rank.**
   `Q̂` is an 11 × 11 matrix built from 6 observations. After the mean is subtracted, only
   5 of them are independent, so

   ```math
   \text{rank}(\hat{Q}) \;\le\; L - 1 \;=\; 5 \;<\; 11
   ```

2. **So `Q̂` is singular: it has no inverse.**
   The solution formula needs `Q̂⁻¹`, and it cannot be computed.

3. **In portfolio terms, some trades look riskless.**
   A singular `Q̂` means there are long-short portfolios `d` with `Q̂d = 0`. Their return
   did not vary at all during those 6 months, so their estimated variance is exactly zero.

4. **The optimizer wants an infinite amount of them.**
   If `μ̂` says such a trade made money, adding `t` units of it raises the expected return
   and adds no estimated risk:

   ```math
   \hat{Q}d = 0,\;\; e^\top d = 0,\;\; \hat{\mu}^\top d > 0
   \quad\Longrightarrow\quad
   f(x + t\,d) \;=\; f(x) + t\,\hat{\mu}^\top d \;\longrightarrow\; \infty
   ```

   where `f` is the MVO objective. There is no best portfolio.

5. **A solver still prints numbers.**
   They are decided by its numerical tolerance, not by anything in the market.

## Challenge 3: even when it can be solved, the answer is fragile

Now suppose there is enough data for `Q̂` to be inverted. The problem from Challenge 2 does
not disappear. It comes back in a milder form: a solution exists, but a small change in the
inputs moves it a long way.

**A two-asset example.** Take two assets with the same volatility `σ` and correlation `ρ`.
Solving MVO by hand gives the gap between their weights:

```math
x_1 - x_2 \;=\; \frac{\mu_1 - \mu_2}{2\lambda\,\sigma^2\,(1 - \rho)}
```

Read it as: the difference in expected returns, divided by how different the two assets are.

- If the assets are unrelated (`ρ = 0`), the denominator is large and the position is small.
- If they move almost together (`ρ → 1`), the denominator goes to zero. The two assets are
  near substitutes, so "buy one, short the other" looks almost riskless, and the optimizer
  takes a huge position on any difference in their expected returns.

With `λ = 2.5`, `σ = 5%` per month, and a difference in expected return of only 0.1% per
month:

| Correlation `ρ` | Weight gap `x₁ − x₂` |
|---|---|
| 0 | 0.08 |
| 0.9 | 0.8 |
| 0.99 | 8 |
| 0.999 | 80 |

The forecast is the same in every row. Only the correlation changes, and the position grows
a thousandfold.

**Why this bites in practice.** Two things happen at the same time.

1. **The assets are correlated, so `Q̂` is close to singular.**
   Sector ETFs all move with the market. With 11 assets it is not only pairs that matter:
   any combination of assets that nearly replicates another has the same effect, and in a
   short sample such combinations also appear by chance.

2. **The return forecast is very noisy.**
   The standard error of an estimated mean is

   ```math
   \text{SE}(\hat{\mu}_i) = \frac{\sigma_i}{\sqrt{L}}
   ```

   It shrinks only with the square root of the amount of data.

Put together: the optimizer takes large positions (point 1) on differences in expected
return that are mostly estimation noise (point 2). The noise is different every month, so
the positions keep flipping.

In this study's data:

- The average pairwise correlation of the 11 ETFs is 0.63.
- The average ETF has a monthly mean return of 1.2% and a monthly volatility of 5.8%. With
  `L = 24` the standard error of that mean is 5.8% / √24 ≈ 1.2%, as large as the mean itself.
- Raising one ETF's estimated mean by 0.1% per month, one-twelfth of that standard error,
  changes the MVO weights by a total of 2.0 at `L = 24`: trades worth 200% of the
  portfolio. At `L = 12` the figure is about 21. (Medians over all windows and assets.)

**The general measure: the condition number.** For more than two assets, the role of
`1 − ρ` is played by the eigenvalues of `Q̂`:

```math
\kappa(\hat{Q}) = \frac{\lambda_{\max}(\hat{Q})}{\lambda_{\min}(\hat{Q})}
```

- A small `λ_min` means some portfolio has an estimated variance of almost zero. That is the
  "almost riskless" trade from the example.
- The larger `κ` is, the more `Q̂⁻¹` amplifies errors. For the step `Q̂⁻¹μ̂` at the heart of
  the solution, a 1% error in the inputs can become an error of up to `κ`% in the result.
- `κ = ∞` means `Q̂` is singular. That is Challenge 2.

So Challenge 2 is the extreme case of Challenge 3, not a separate problem. The three
estimation windows used in this study cover both:

| Window | State of `Q̂` | Which challenge |
|---|---|---|
| `L = 6` | singular, `κ = ∞` | Challenge 2: no solution |
| `L = 12` | invertible, close to singular | Challenge 3, severe |
| `L = 24` | invertible, better conditioned | Challenge 3, milder |

## Five remedies for the three challenges

The other models in this study are the standard remedies for these challenges. They
intervene at different points:

- two of them change the **objective** (robust MVO, with two shapes of uncertainty set),
- one changes the **return forecast** (Black–Litterman),
- two stop **using the inputs** (risk parity, equal weight).

One thing to keep in mind: **none of the five repairs `Q̂` itself.** Each one works around
the fragile inverse in a different way. Whether that is enough is what the study tests.

### Robust MVO: plan for the worst case

**Idea.** Admit that `μ̂` is uncertain. Say that the true `μ` lies somewhere in a set `U`
around `μ̂`, and choose the portfolio that does best if the worst `μ` in that set occurs:

```math
\max_{x}\; \min_{\mu \in U}\; \mu^\top x \;-\; \lambda\, x^\top \hat{Q}\, x \qquad \text{subject to} \qquad e^\top x = 1
```

The inner minimum can be worked out by hand. It turns into a penalty subtracted from the
MVO objective, and the shape of `U` decides what the penalty looks like.

**Version A: ellipsoidal set** (`robust_ellipsoidal`)

```math
\max_{x}\; \hat{\mu}^\top x \;-\; \rho\sqrt{x^\top S\, x} \;-\; \lambda\, x^\top \hat{Q}\, x,
\qquad S = \hat{Q}/L
```

`S` is the covariance of the estimation error in `μ̂`, and `ρ` sets the confidence level.

- **Addresses:** Challenge 1. The penalty is large for portfolios whose expected return is
  poorly estimated.
- **Leaves open:** Challenges 2 and 3. The penalty is built from `Q̂` itself, so a trade
  that `Q̂` rates as riskless is also rated as free of estimation error. In fact the
  solution is the plain MVO solution for a larger `λ`: the same bets, scaled down.

**Version B: box set** (`robust_box`)

```math
\max_{x}\; \hat{\mu}^\top x \;-\; \delta^\top |x| \;-\; \lambda\, x^\top \hat{Q}\, x,
\qquad \delta_i = z\,\sqrt{\hat{Q}_{ii}/L}
```

`δᵢ` is the half-width of a confidence interval around `μ̂ᵢ`.

- **Addresses:** Challenge 1, and Challenges 2 and 3 as a side effect. Every position, long
  or short, pays a charge `δᵢ`. A trade is worth taking only if its expected gain is larger
  than that charge, whatever `Q̂` says about its risk.
- **Leaves open:** `Q̂` is untouched. With short windows `δ` is large, so the model may hold
  back from good positions as well as bad ones. The confidence level has to be chosen.

### Black–Litterman: replace the forecast

**Idea.** Do not use the sample mean. Start from the returns that would make a reference
"market" portfolio `w` optimal, then adjust them toward the investor's views
(`black_litterman`):

```math
\Pi = \delta\, \hat{Q}\, w,
\qquad
\mu_{BL} = \left[(\tau\hat{Q})^{-1} + P^\top \Omega^{-1} P\right]^{-1}
           \left[(\tau\hat{Q})^{-1}\Pi + P^\top \Omega^{-1} q\right]
```

`P` and `q` state the views, and `Ω` says how confident they are. `μ_BL` then replaces `μ̂`
in ordinary MVO.

- **Addresses:** Challenge 1 for the mean, which is the noisiest input. `Π` is built from
  `Q̂` and the reference weights, not from average past returns, so it cannot promise a
  profit on a trade that `Q̂` rates as riskless. That removes the incentive behind
  Challenge 2.
- **Leaves open:** Challenge 3 in part. The last step is still MVO with `Q̂⁻¹`, so a view is
  amplified in the same way a noisy mean would be. The model also needs inputs of its own:
  reference weights, views, and how much to trust them.

### Risk parity: stop forecasting returns

**Idea.** Ignore `μ` altogether. Hold every asset, sized so that each one contributes the
same amount of risk to the portfolio (`risk_parity`):

```math
x_i\,(\hat{Q}x)_i = x_j\,(\hat{Q}x)_j \;\;\text{for all } i, j,
\qquad x \ge 0, \qquad e^\top x = 1
```

- **Addresses:** all three. There is no `μ̂` to be wrong about. The portfolio is long-only,
  so it cannot build the long-short trades behind Challenges 2 and 3 at all.
- **Leaves open:** it still depends on `Q̂`, though only mildly. It gives up every piece of
  information about expected returns, and it is not designed to maximize return for the
  risk taken.

### Equal weight: stop estimating

**Idea.** Put `1/n` in each asset (`equal_weight`):

```math
x_i = \frac{1}{n}
```

- **Addresses:** all three, by not taking part. Nothing is estimated, so nothing can be
  estimated badly.
- **Leaves open:** it ignores risk as well as return. If one asset is much more volatile
  than the rest, it dominates the portfolio's risk. It serves as the benchmark that any
  estimated model has to beat.

### Summary: what each remedy should do, on paper

| Remedy | What it changes | Challenge 1: noisy inputs | Challenge 2: `Q̂` singular | Challenge 3: sensitivity | Main drawback |
|---|---|---|---|---|---|
| Robust, ellipsoidal | objective: penalty `ρ·√(xᵀSx)` | Yes | No | No: same bets, scaled down | penalty is blind where `Q̂` is blind |
| Robust, box | objective: penalty `δᵀ\|x\|` | Yes | Mostly: every position is charged | Partly | can be too cautious when data is short |
| Black–Litterman | forecast: `μ̂` → `μ_BL` | Yes, for `μ` | Mostly: no reward for riskless trades | Partly: `Q̂⁻¹` still used | needs views and reference weights |
| Risk parity | drops `μ`, long-only | Avoided | Avoided | Mostly avoided | ignores expected returns |
| Equal weight | drops all estimation | Avoided | Avoided | Avoided | ignores risk as well |

This table is the expectation. The rest of the study checks it against data.

## What this study asks

Every one of these remedies is reasonable on paper. The goal here is not to find the
strategy with the highest Sharpe ratio. It is to make the effect of estimation error
visible and to see which remedies hold up when the three challenges are real.

1. **How severe is the failure at realistic sample sizes?** Does it disappear once there is
   enough data for `Q̂` to be invertible?
2. **Which remedies contain it?** Does the table above survive contact with data?
3. **What does stability cost?** Once a portfolio is well-behaved, how much does it give up
   in turnover and transaction costs?

This is worth knowing for two reasons.

- **It is the gap between the model as taught and the model as used.** Almost no
  practitioner runs MVO unmodified. Seeing exactly where it breaks explains what they do
  instead.
- **It is not specific to this dataset.** Any pipeline that feeds estimated inputs into an
  optimizer has the same structure: factor portfolios, risk budgeting, hedging. Most have
  far more assets per observation than the 11 here, so the problem is worse.

## How the study is set up

**The experiment.** Each month, estimate `μ̂` and `Q̂` from the previous `L` months only,
build every portfolio, hold it for one month, record the realized return net of transaction
costs, and roll forward. Month `t` is never used to choose the portfolio that is evaluated
on month `t`.

**The dial.** The estimation window `L` is the control variable. The three values
`L = 6`, `12` and `24` are the three cases from Challenges 2 and 3: `Q̂` singular, close
to singular, and better conditioned.

**The strategies.** Plain MVO, plus the five remedies described above. Both robust models
are run at 90%, 95% and 99% confidence, which gives ten strategies in total.

MVO is deliberately left unregularized. Shrinking `Q̂` or capping leverage would give usable
numbers and remove the thing being studied.

**Details.**

| | |
|---|---|
| Universe | XLK, XLF, XLE, XLY, XLP, XLV, XLI, XLB, XLU, XLRE, XLC |
| Data | Monthly returns from end-of-month adjusted prices, 2019-01 to 2025-12 (84 months) |
| Transaction costs | 10 bps on turnover `‖x_t − x_{t−1}‖₁`; the first month pays the full cost of building the position |
| Risk aversion | `λ = 2.5` for every optimizer, so differences come from the model and not the tuning |
| Robust confidence levels | 90%, 95%, 99% for both uncertainty sets |
| Black–Litterman view | XLK outperforms XLU by 0.25% per month, `τ = 0.05`, equal-weight market proxy |

The models are pure functions `(μ̂, Q̂) → weights` in [src/models.py](src/models.py); the
rolling driver is [src/backtest.py](src/backtest.py).

## What the data show

The first half made its claims on paper. This half checks them. Three measurements are used,
and each one is tied to something from the first half:

| Measurement | What it shows | Tied to |
|---|---|---|
| Gross leverage `‖x‖₁` | whether the optimizer is piling into trades that look riskless. 1.0 means fully invested with no short positions | Challenge 2 |
| Monthly turnover `‖x_t − x_{t−1}‖₁` | how far the weights move from one month to the next when the inputs change a little. 1.0 means trades worth 100% of the portfolio | Challenge 3 |
| Net Sharpe ratio | whether the portfolio pays after transaction costs | the price of a remedy |

### Check 1: are the challenges real?

Plain MVO, with nothing done about the three challenges:

| Window | rank of `Q̂` (of 11) | median `κ(Q̂)` | Average gross leverage | Average monthly turnover | Out of sample |
|---|---|---|---|---|---|
| `L = 6` | 5 | ∞ (singular) | 1.3 × 10¹⁰ | 9.4 × 10⁹ | wealth passes through zero |
| `L = 12` | 11 | 2.7 × 10⁴ | 2,779 | 3,854 | wealth passes through zero |
| `L = 24` | 11 | 376 | 39.5 | 16.5 | loses all capital; net Sharpe −0.08 (annualized) |

- **`L = 6` is Challenge 2.**
  `Q̂` has rank 5. The trades it rates as riskless form a five-dimensional space, and in the
  median window 32% of the norm of `μ̂` points into it. The problem has no solution. The
  10¹⁰ figure comes from the 10⁻¹² eigenvalue floor that lets the solver return anything at
  all; it describes the solver setup, not the market.
- **`L = 12` and `L = 24` are Challenge 3.**
  `Q̂` is invertible and the problem is well-defined. Leverage is still in the thousands at
  `L = 12`. At `L = 24` the portfolio is levered 40×, trades 16 times its own value every
  month, loses 80% of its capital within nine months and all of it by month 25.
- **More data shrinks the problem without removing it.**
  `κ` falls from ∞ to 2.7 × 10⁴ to 376, and leverage falls with it, but never to a usable
  level. As Challenge 3 argued, a singular `Q̂` is the extreme end of a continuous problem.

### Check 2: did each remedy do what it should?

**Position size** (average gross leverage), the test for Challenge 2:

| Strategy | `L = 6` | `L = 12` | `L = 24` |
|---|---|---|---|
| MVO | 1.3 × 10¹⁰ | 2,779 | 39.5 |
| Robust, ellipsoidal (95%) | 1.3 × 10¹⁰ | 2,468 | 11.0 |
| Robust, box (95%) | 1.10 | 1.00 | 1.00 |
| Black–Litterman | 2.13 | 7.75 | 2.38 |
| Risk parity | 1.00 | 1.00 | 1.00 |
| Equal weight | 1.00 | 1.00 | 1.00 |

**Stability of the weights** (average monthly turnover), the test for Challenge 3:

| Strategy | `L = 6` | `L = 12` | `L = 24` |
|---|---|---|---|
| MVO | 9.4 × 10⁹ | 3,854 | 16.5 |
| Robust, ellipsoidal (95%) | 9.4 × 10⁹ | 3,602 | 5.38 |
| Robust, box (95%) | 0.99 | 0.67 | 0.39 |
| Black–Litterman | 1.59 | 7.38 | 0.75 |
| Risk parity | 0.24 | 0.10 | 0.05 |
| Equal weight | 0.01 | 0.01 | 0.02 |

![Average gross leverage by strategy, L = 12](figures/instability_L12.png)

Set against the summary table from the first half:

| Remedy | Expected on paper | Observed | Verdict |
|---|---|---|---|
| Robust, ellipsoidal | no help with Challenge 2; the same bets scaled down under Challenge 3 | identical to MVO at `L = 6`; leverage 11 instead of 40 at `L = 24` | as expected: it fails |
| Robust, box | mostly contains both; may be too cautious on short data | leverage about 1 at every window; lowest Sharpe of the stable group at `L = 6` | as expected |
| Black–Litterman | bounded, but still sensitive | leverage 2 to 8; turnover as high as 7.4 | as expected |
| Risk parity | avoids both | leverage 1.00; turnover at most 0.24 | as expected |
| Equal weight | avoids both | leverage 1.00; turnover near zero | as expected |

Why each one came out this way:

- **Ellipsoidal robustness measures with the broken instrument.**
  Its penalty uses `S = Q̂/L`, the same matrix rescaled. Wherever `Q̂` sees no risk, `S` sees
  no estimation error, so the penalty vanishes in exactly the directions where it is needed.
  At `L = 24` it behaves like plain MVO with a risk aversion about five times larger: the
  same bets, smaller.
- **Box robustness charges for every position.**
  `δᵀ|x|` is strictly positive in every direction (the smallest `δᵢ` across all `L = 6`
  windows is 0.0086). What is motivated as a robustness adjustment to the mean acts as a
  leverage constraint. This is the mechanism behind the norm-constrained portfolios of
  DeMiguel, Garlappi, Nogales and Uppal (2009) and the "wrong constraints help" result of
  Jagannathan and Ma (2003).
- **Black–Litterman removes the incentive.**
  At `L = 6` the share of `μ_BL` lying in the riskless space is of order 10⁻¹¹, against 32%
  for `μ̂`. The optimizer is never told that a riskless trade is profitable. But the last
  step is still MVO, and the turnover shows it.
- **Risk parity and equal weight step outside the problem.**
  One discards `μ` and holds only long positions; the other estimates nothing.

One rule explains the split: **a remedy works when it still restrains a trade that `Q̂`
rates as riskless.** A penalty, a prior or a constraint can all do that. A correction built
from `Q̂` alone cannot.

### Check 3: what does a remedy cost?

Net-of-cost annualized Sharpe ratio:

| Strategy | `L = 6` | `L = 12` | `L = 24` |
|---|---|---|---|
| Equal weight | 0.82 | 0.77 | **0.89** |
| Risk parity | **0.83** | 0.78 | 0.83 |
| Black–Litterman | 0.67 | **0.79** | 0.79 |
| Robust, box (95%) | 0.29 | 0.58 | 0.58 |
| Robust, ellipsoidal (95%) | n/a | n/a | 0.07 |
| MVO | n/a | n/a | −0.08 |

"n/a" marks runs where wealth crosses zero, so return statistics have no economic meaning.

![Cumulative wealth, L = 24, net of transaction costs](figures/cumwealth_L24.png)

- **The ranking runs in reverse order of sophistication.**
  The fewer quantities a strategy estimates, the better it does net of costs. This
  reproduces the 1/N result of DeMiguel, Garlappi and Uppal (2009) on a different universe
  and period.
- **Black–Litterman pays in turnover.**
  At `L = 12` it has the highest terminal wealth of any strategy (3.20), turns its portfolio
  over 7.4 times a month, and gives up 0.74% of return per month to a 10 bps cost.
- **Box robustness pays in caution.**
  Its charge `δ` is largest when data is shortest, which is when it holds back the most:
  a Sharpe ratio of 0.29 at `L = 6` against 0.82 for equal weight.

## Conclusion

**Answers to the three questions.**

1. **How severe is the failure?**
   Severe at every window tested. More data makes it smaller and never makes it go away.
2. **Which remedies contain it?**
   Box robustness, Black–Litterman, risk parity and equal weight. Ellipsoidal robustness
   does not. The on-paper analysis predicted all five outcomes.
3. **What does stability cost?**
   The simplest portfolios are also the cheapest to run. Among the rest, Black–Litterman
   pays in turnover and box robustness pays in caution.

**The bigger picture.** Estimation error is not a small perturbation of the textbook
problem. At realistic sample sizes it changes which problem is being solved: the optimizer
is handed a risk model with blind spots and a return forecast that points into them.
Whatever is added to contain that, whether a penalty, a prior or a constraint, ends up
determining the portfolio more than the mean-variance objective does. The choice of
regularizer is the model.

Three things follow from that.

- **"Robust" describes an uncertainty set, not a portfolio.** Two models with the same
  motivation and the same confidence level behave in opposite ways, because one penalty is
  degenerate where the other is not. The shape of the set matters more than its size: no
  confidence level rescues the ellipsoidal model at short windows, and the box model stays
  bounded at all three.
- **Check conditioning before reading performance.** Gross leverage and the condition number
  of `Q̂` flagged the failure before a single return was realized. A backtest that reports
  only Sharpe ratios would show the blown-up runs as noisy rather than as ill-posed.
- **A simple portfolio is a benchmark to beat, not a fallback.** If a method cannot
  outperform 1/N after costs, its extra machinery is spending estimation risk without
  earning anything for it.

## Limitations

- **One sample, short tracks.** 60 to 78 out-of-sample months. The standard error of an
  annualized Sharpe ratio over 60 months is roughly 0.45 (i.i.d. approximation of Lo, 2002),
  so the ranking among equal weight, risk parity and Black–Litterman is not statistically
  distinguishable. The stability results differ by orders of magnitude and do not depend on
  that ranking.
- **`λ` is fixed at 2.5.** A larger `λ` scales down the speculative part of the MVO
  position at `L = 12` and `L = 24`; it does not make the `L = 6` problem bounded. A
  sensitivity sweep is not yet done.
- **Black–Litterman uses an equal-weight market proxy** and a single illustrative view, not
  capitalization weights.
- **Blown-up magnitudes are solver-dependent.** The direction of the result is not; the
  exact exponent is.
- **Sample period** includes the 2020 crash and the 2022 rate cycle; no sub-period analysis.

## Next steps

1. Add Ledoit–Wolf shrinkage and an explicit leverage cap to MVO, to test the conclusion
   directly: if it is right, repairing `Q̂` should also repair the ellipsoidal model.
2. Sweep `λ ∈ {1, 2.5, 5, 10, 25}` and report leverage, Sharpe and turnover against it.
3. Rebuild the Black–Litterman prior from sector market-cap weights.
4. Sub-period robustness (excluding 2020; splitting at 2022).

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run.py
```

`python run.py` rebuilds every table in `results/` and every figure in `figures/` from the
cached backtests, so it reproduces the numbers above exactly. `python run.py --force`
re-solves every model from scratch; the stable strategies reproduce, while the blown-up
MVO and ellipsoidal magnitudes may shift with the solver path. `python run.py --download`
refreshes prices first.

Tested on Python 3.9 with the pinned versions in [requirements.txt](requirements.txt).

### Tests

```bash
pytest
```

The suite checks the claims the results rest on: no look-ahead (perturbing a future month
leaves all earlier weights unchanged), the budget constraint `eᵀx = 1`, equalized risk
contributions in risk parity, the Black–Litterman posterior moving toward the view, the
identity `net = gross − c·turnover`, and each performance-metric formula.

### Layout

```
src/
  data.py        download adjusted prices -> data/monthly_returns.csv
  models.py      the six models as pure (mu, Q) -> weights functions
  backtest.py    rolling out-of-sample driver, turnover and transaction costs
  metrics.py     Sharpe, Sortino, max drawdown, turnover, cost-impact tables
  figures.py     cumulative wealth, rolling Sharpe, weight heatmaps, frontier, leverage
run.py           one command: backtest -> results/ -> figures/
tests/           pytest suite
data/            monthly_returns.csv
results/         cached backtests and performance / turnover / tc_impact tables
figures/         all figures
```

On macOS with numpy 2.0 and the Accelerate BLAS backend, `matmul` can emit spurious
`RuntimeWarning`s on finite results. They are harmless and filtered in `pytest.ini`.

## References

- Markowitz, H. (1952). Portfolio selection. *Journal of Finance*, 7(1).
- Michaud, R. (1989). The Markowitz optimization enigma: Is "optimized" optimal? *Financial Analysts Journal*, 45(1).
- Best, M., & Grauer, R. (1991). On the sensitivity of mean-variance-efficient portfolios to changes in asset means. *Review of Financial Studies*, 4(2).
- Chopra, V., & Ziemba, W. (1993). The effect of errors in means, variances, and covariances on optimal portfolio choice. *Journal of Portfolio Management*, 19(2).
- Black, F., & Litterman, R. (1992). Global portfolio optimization. *Financial Analysts Journal*, 48(5).
- Lo, A. (2002). The statistics of Sharpe ratios. *Financial Analysts Journal*, 58(4).
- Jagannathan, R., & Ma, T. (2003). Risk reduction in large portfolios: Why imposing the wrong constraints helps. *Journal of Finance*, 58(4).
- Goldfarb, D., & Iyengar, G. (2003). Robust portfolio selection problems. *Mathematics of Operations Research*, 28(1).
- Tütüncü, R., & Koenig, M. (2004). Robust asset allocation. *Annals of Operations Research*, 132.
- Ledoit, O., & Wolf, M. (2004). Honey, I shrunk the sample covariance matrix. *Journal of Portfolio Management*, 30(4).
- DeMiguel, V., Garlappi, L., & Uppal, R. (2009). Optimal versus naive diversification: How inefficient is the 1/N portfolio strategy? *Review of Financial Studies*, 22(5).
- DeMiguel, V., Garlappi, L., Nogales, F., & Uppal, R. (2009). A generalized approach to portfolio optimization: Improving performance by constraining portfolio norms. *Management Science*, 55(5).
- Maillard, S., Roncalli, T., & Teïletche, J. (2010). The properties of equally weighted risk contribution portfolios. *Journal of Portfolio Management*, 36(4).
- Cornuéjols, G., Peña, J., & Tütüncü, R. (2018). *Optimization Methods in Finance* (2nd ed.). Cambridge University Press.

*This is a research exercise on historical data, not investment advice.*
