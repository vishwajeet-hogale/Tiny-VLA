"""Pairwise verdicts: is policy A really better than policy B, or is it noise?

Every policy is evaluated on the same seeds, so the comparison is *paired*: seed 1042 is the same
starting layout for both. Seeds where both succeed or both fail say nothing about which is better;
only the disagreements do. McNemar's exact test asks: among the n seeds where exactly one policy
succeeded, is the split between "only A" and "only B" more lopsided than a fair coin would give?

Success rates get a Wilson 95% confidence interval, which behaves well near 0% and 100% (unlike
p +/- 1.96 * sqrt(p(1-p)/n)).

As a library (used by compare_models.py):
    from compare import wilson_interval, mcnemar_exact, verdict
From the command line, with two per-episode CSVs that have `seed` and `success` columns:
    python scripts/compare.py outputs/part6/episodes_A.csv outputs/part6/episodes_B.csv
"""

import csv
import math
import sys


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a success rate (Wilson score interval)."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, center - half), min(1.0, center + half))


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided p-value: probability of a split at least this lopsided if A and B were equal.

    Under "no difference", each disagreement is a fair coin flip, so only_a ~ Binomial(n, 0.5).
    """
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_counts(success_a: dict[int, bool], success_b: dict[int, bool]) -> dict:
    """Count agreements and disagreements over the seeds both policies ran."""
    seeds = sorted(set(success_a) & set(success_b))
    counts = {"both": 0, "only_a": 0, "only_b": 0, "neither": 0, "n": len(seeds)}
    for seed in seeds:
        a, b = success_a[seed], success_b[seed]
        key = "both" if a and b else "only_a" if a else "only_b" if b else "neither"
        counts[key] += 1
    return counts


def verdict(name_a: str, success_a: dict[int, bool], name_b: str, success_b: dict[int, bool],
            alpha: float = 0.05) -> str:
    """One line: which policy is better on these seeds, and how confident we can be."""
    c = paired_counts(success_a, success_b)
    p = mcnemar_exact(c["only_a"], c["only_b"])
    rate_a = (c["both"] + c["only_a"]) / c["n"]
    rate_b = (c["both"] + c["only_b"]) / c["n"]
    detail = (f"{rate_a:.0%} vs {rate_b:.0%} on {c['n']} seeds; only {name_a}: {c['only_a']}, "
              f"only {name_b}: {c['only_b']}; McNemar p = {p:.3g}")
    if p >= alpha:
        return f"{name_a} vs {name_b}: no significant difference ({detail})"
    better = name_a if c["only_a"] > c["only_b"] else name_b
    return f"{name_a} vs {name_b}: {better} is better ({detail})"


def load_successes(path: str) -> dict[int, bool]:
    return {int(r["seed"]): r["success"] in ("True", "true", "1") for r in csv.DictReader(open(path))}


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    a, b = load_successes(sys.argv[1]), load_successes(sys.argv[2])
    for name, s in [(sys.argv[1], a), (sys.argv[2], b)]:
        lo, hi = wilson_interval(sum(s.values()), len(s))
        print(f"{name}: {sum(s.values())}/{len(s)} = {sum(s.values()) / len(s):.0%}  (95% CI {lo:.0%}-{hi:.0%})")
    print(verdict("A", a, "B", b))
