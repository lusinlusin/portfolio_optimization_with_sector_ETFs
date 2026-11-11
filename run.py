"""One-command reproduction of every result and figure in the study.

Pipeline
--------
1. (optional) download daily adjusted prices -> data/monthly_returns.csv
2. run the rolling-horizon backtest for L in {6, 12, 24}  (cached to results/*.pkl)
3. write the three summary tables to results/
       performance.csv   (metrics.performance_table)
       turnover.csv      (metrics.turnover_table)
       tc_impact.csv     (metrics.tc_impact_table)
4. regenerate every figure into figures/
5. print the programmatic answers to the study's comparison questions

Usage
-----
    python run.py              # use existing data/, reuse cached backtests if present
    python run.py --download   # re-download prices first (needs internet)
    python run.py --force      # re-solve every model even if results/*.pkl exists

By default the backtest is loaded from results/backtest_L{L}.pkl when present, so a
fresh checkout reproduces the *exact* numbers cited in README.md (the blown-up
MVO / ellipsoidal magnitudes at short windows are solver-path dependent, so only
``--force`` re-solves them).  All design choices (lambda, transaction-cost rate,
window lengths, BL view, x_0 convention) live in src/backtest.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--download", action="store_true",
                    help="re-download prices into data/monthly_returns.csv first")
    ap.add_argument("--force", action="store_true",
                    help="re-solve the backtest even if results/*.pkl is cached")
    args = ap.parse_args()

    # Imported here (not at module top) so --help works without the heavy deps.
    from src import data as D
    from src import backtest as BT
    from src import metrics as MT
    from src import figures as FG

    results_dir = ROOT / BT.RESULTS_DIR
    results_dir.mkdir(exist_ok=True)

    # 1. data ----------------------------------------------------------------
    if args.download:
        print(f"[1/5] downloading {len(D.ASSETS)} ETFs {D.START}..{D.END} ...")
        returns = D.download_data(D.ASSETS, D.START, D.END).dropna(how="all")
        (ROOT / "data").mkdir(exist_ok=True)
        returns.to_csv(ROOT / "data" / "monthly_returns.csv")
    print("[1/5] loading monthly returns ...")
    returns = BT.load_returns(str(ROOT / BT.DATA_PATH))
    print(f"      {len(returns)} monthly returns, "
          f"{returns.index[0]:%Y-%m} .. {returns.index[-1]:%Y-%m}, "
          f"{returns.shape[1]} assets")

    # 2. backtest (reuse cache unless --force) -------------------------------
    all_results: dict = {}
    for L in BT.WINDOWS:
        pkl = results_dir / f"backtest_L{L}.pkl"
        if pkl.exists() and not args.force:
            print(f"[2/5] L={L:>2}: loading cached {pkl.name}")
            all_results[L] = BT.load_results(L, outdir=str(results_dir))
        else:
            print(f"[2/5] L={L:>2}: solving all strategies ...")
            res = BT.run_backtest(returns, L)
            all_results[L] = res
            import pickle
            with open(pkl, "wb") as f:
                pickle.dump(res, f)

    # 3. summary tables ------------------------------------------------------
    print("[3/5] writing performance / turnover / tc_impact tables ...")
    MT.performance_table(all_results).to_csv(results_dir / "performance.csv")
    MT.turnover_table(all_results).to_csv(results_dir / "turnover.csv")
    MT.tc_impact_table(all_results).to_csv(results_dir / "tc_impact.csv")

    # 4. figures -------------------------------------------------------------
    print("[4/5] regenerating figures ...")
    FG.save_all_figures(all_results, returns, outdir=str(ROOT / FG.FIG_DIR))

    # 5. discussion answers --------------------------------------------------
    print("[5/5] discussion answers (net, stable strategies only):")
    answers = MT.discussion_answers(all_results)
    for L, a in answers.items():
        print(f"  L={L}:")
        for k, v in a.items():
            print(f"    {k:>22}: {v}")

    print("\nDone. Tables in results/, figures in figures/.")


if __name__ == "__main__":
    main()
