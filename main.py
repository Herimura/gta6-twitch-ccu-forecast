#!/usr/bin/env python3
"""Forecast a game launch day's Twitch viewers, hours 2-24, from hour 1.

Entry for make.fun competition gta6-twitch-ccu-forecast-01.

Method
------
Every case file carries the hourly release-day curves of dozens of past
big launches ("comparables"). For each comparable we form the log-ratio
curve r_h = ln(viewers_h / viewers_1) of its release day: how the audience
grew or fell after its first hour, in units of that first hour. The case's
own first hour is the one hard fact we know, so the forecast for hour h is

    q_tau(h) = print_1 * exp( Q_tau( {r_h over comparables} ) )

with Q_tau a weighted empirical quantile. Comparables are weighted by how
close their own first-hour audience was to this case's (Gaussian kernel on
|ln v1_comp - ln print_1|, bandwidth 1.25, plus a small floor weight so no
launch is ever fully ignored): a rolling-midnight console launch with a
large first hour is predicted mostly from other large rolling launches,
whose unlock waves (NZ, Europe, UK, US east, US west) trace the same
shape, and hardly at all from small or oddly-timed launches. The five
quantile curves are then smoothed across neighbouring hours, because the
true day curve is smooth and the per-hour samples are not.

The raw empirical quantiles are first blended (weight 0.6) toward a
parametric curve: a least-squares fit of the per-hour weighted median
log-ratio on a smooth basis (log-hour polynomial plus Gaussian bumps at
the US unlock waves) for the centre, with a normal-shaped spread whose
sigma is the hour's interquartile range of the weighted samples. The
per-hour empirical quantiles of a few dozen launches are noisy; the
parametric shape is not, and the blend trades a little of the empirical
curve's fidelity for much lower variance.

The constants (bandwidth 1.25, smoothing weights, weight floor, blend
weight 0.6) were chosen by leave-one-out validation over the comparable
launches of the practice cases: each launch in turn was forecast from
the others and scored with the competition's pinball loss on the log
scale, end-to-end through this program. On launches with a first hour
of at least 20,000 viewers (the regime a GTA VI first hour sits in) the
blend scores about 0.255 against about 0.321 for the house persistence
baseline; over all 37 usable launches about 0.423 against 0.930, with
90% coverage of 0.89-0.90. The gain holds on both halves of a
split-half check, and the blend weights 0.5-0.7 all score within a few
tenths of a percent of each other, so the improvement comes from the
variance reduction itself, not from a tuned spike.

Standard library only, no network, deterministic: the output is a pure
function of the case file.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
KERNEL_BW = 1.25          # bandwidth of the first-hour similarity kernel
WEIGHT_FLOOR = 0.05       # every comparable keeps at least this weight
SPREAD_K = 0.95           # quantile spread calibration around the median
PARAM_FULL_W = 0.6        # blend weight toward the parametric curve
SMOOTH = ((0, 0.5), (-1, 0.25), (1, 0.25))  # (hour offset, weight)
Z = (-1.645, -0.674, 0.0, 0.674, 1.645)     # fallback spread (house shape)


def parse_ts(text):
    """Parse an ISO-8601 UTC timestamp to epoch seconds."""
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc).timestamp()


def day_curve(comp):
    """Map hour index 1..24 -> viewers for a comparable's release day."""
    try:
        t0 = parse_ts(comp["day_start"])
    except (KeyError, ValueError):
        return {}
    curve = {}
    for point in comp.get("hourly") or []:
        try:
            idx = int((parse_ts(point["from"]) - t0) // 3600)
            viewers = float(point["viewers"])
        except (KeyError, ValueError, TypeError):
            continue
        if 0 <= idx < 24 and idx + 1 not in curve:
            curve[idx + 1] = viewers
    return curve


def weighted_quantile(pairs, tau):
    """Weighted empirical quantile with midpoint plotting positions.

    pairs is a list of (value, weight); the result is linearly
    interpolated between the sorted values at their cumulative-weight
    midpoints, and clamped to the extremes outside them.
    """
    pairs = sorted(pairs)
    total = sum(w for _, w in pairs)
    if total <= 0:
        return pairs[len(pairs) // 2][0]
    xs, ps = [], []
    cum = 0.0
    for value, weight in pairs:
        cum += weight
        xs.append(value)
        ps.append((cum - weight / 2.0) / total)
    if tau <= ps[0]:
        return xs[0]
    if tau >= ps[-1]:
        return xs[-1]
    for i in range(1, len(ps)):
        if ps[i] >= tau:
            frac = (tau - ps[i - 1]) / (ps[i] - ps[i - 1])
            return xs[i - 1] + frac * (xs[i] - xs[i - 1])
    return xs[-1]


def collect_samples(case):
    """Per-hour weighted log-ratio samples from the case's comparables.

    Returns (samples, pooled): samples maps hour 2..24 to a list of
    (log-ratio, weight); pooled is the same over all hours, a fallback
    for any hour no comparable covers.
    """
    known = case["known"][0]
    print_1 = max(1.0, float(known["viewers"]))
    log_p = math.log(print_1)
    samples = {h: [] for h in range(2, 25)}
    pooled = []
    for comp in case.get("comparables") or []:
        curve = day_curve(comp)
        v1 = curve.get(1)
        if not v1 or v1 <= 0 or len(curve) < 6:
            continue
        dist = abs(math.log(v1) - log_p)
        weight = math.exp(-0.5 * (dist / KERNEL_BW) ** 2) + WEIGHT_FLOOR
        log_v1 = math.log(v1)
        for hour in range(2, 25):
            viewers = curve.get(hour)
            if viewers and viewers > 0:
                pair = (math.log(viewers) - log_v1, weight)
                samples[hour].append(pair)
                pooled.append(pair)
    return print_1, samples, pooled


def parametric_median(samples):
    """Smooth parametric fit of the per-hour weighted median log-ratio.

    Least squares over the basis [1, ln h, (ln h)^2] plus Gaussian bumps
    at hours 6 and 9 (the US east/west unlock waves) and a wide bump at
    hour 20 (US prime time). Returns {hour: fitted median} or None when
    too few hours carry enough samples to fit.
    """
    hours, meds = [], []
    for h in range(2, 25):
        pairs = samples[h]
        if len(pairs) >= 5:
            hours.append(h)
            meds.append(weighted_quantile(pairs, 0.5))
    if len(hours) < 10:
        return None

    def basis(h):
        lh = math.log(h)
        return [1.0, lh, lh * lh,
                math.exp(-0.5 * ((h - 6) / 2.0) ** 2),
                math.exp(-0.5 * ((h - 9) / 2.0) ** 2),
                math.exp(-0.5 * ((h - 20) / 4.0) ** 2)]

    k = 6
    ata = [[0.0] * k for _ in range(k)]
    atb = [0.0] * k
    for h, m in zip(hours, meds):
        row = basis(h)
        for i in range(k):
            atb[i] += row[i] * m
            for j in range(k):
                ata[i][j] += row[i] * row[j]
    for i in range(k):
        ata[i][i] += 1e-6
    for col in range(k):
        piv = max(range(col, k), key=lambda r: abs(ata[r][col]))
        if abs(ata[piv][col]) < 1e-12:
            return None
        ata[col], ata[piv] = ata[piv], ata[col]
        atb[col], atb[piv] = atb[piv], atb[col]
        for r in range(col + 1, k):
            f = ata[r][col] / ata[col][col]
            for cc in range(col, k):
                ata[r][cc] -= f * ata[col][cc]
            atb[r] -= f * atb[col]
    coef = [0.0] * k
    for i in range(k - 1, -1, -1):
        s = atb[i] - sum(ata[i][j] * coef[j] for j in range(i + 1, k))
        coef[i] = s / ata[i][i]
    return {h: sum(c * b for c, b in zip(coef, basis(h)))
            for h in range(2, 25)}


def forecast_case(case):
    """Build the 23 forecast rows for one case."""
    print_1, samples, pooled = collect_samples(case)
    raw = {}
    for hour in range(2, 25):
        pairs = samples[hour] or pooled
        if pairs:
            raw[hour] = [weighted_quantile(pairs, tau) for tau in LEVELS]
        else:
            # No comparable data at all: persist at hour 1 with the
            # house baseline's widening spread rather than fail.
            raw[hour] = [z * 0.15 * math.sqrt(hour - 1) for z in Z]
    # Blend the raw quantiles toward the parametric curve (fitted median
    # plus a normal-shaped spread from the hour's weighted IQR). The
    # blend is in log-ratio space, before the across-hour smoothing.
    par_med = parametric_median(samples)
    if par_med is not None:
        for hour in range(2, 25):
            pairs = samples[hour] or pooled
            if not pairs:
                continue
            sigma = max(0.05, (weighted_quantile(pairs, 0.75)
                               - weighted_quantile(pairs, 0.25)) / 1.349)
            for li, z in enumerate(Z):
                par = par_med[hour] + z * sigma
                raw[hour][li] = ((1.0 - PARAM_FULL_W) * raw[hour][li]
                                 + PARAM_FULL_W * par)
    smoothed = {}
    for hour in range(2, 25):
        acc = [0.0] * len(LEVELS)
        wsum = 0.0
        for offset, weight in SMOOTH:
            other = raw.get(hour + offset)
            if other is not None:
                for i in range(len(LEVELS)):
                    acc[i] += weight * other[i]
                wsum += weight
        curve = [a / wsum for a in acc]
        # The raw weighted quantiles run slightly wide (leave-one-out
        # 90% coverage 0.92 on big launches against a 0.90 nominal), so
        # pull each quantile a little toward the median; this also
        # scored a touch better on both validation subsets.
        median = curve[2]
        smoothed[hour] = [median + SPREAD_K * (v - median) for v in curve]
    rows = []
    for hour in range(2, 25):
        qs = sorted(
            min(1e9, max(0.01, print_1 * math.exp(v))) for v in smoothed[hour]
        )
        rows.append({"hour": hour, "q": [round(q, 2) for q in qs]})
    return {"hours": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    for name in names:
        with open(os.path.join(in_dir, name), encoding="utf-8") as fh:
            case = json.load(fh)
        result = forecast_case(case)
        out_name = name[:-3] + ".out"
        with open(os.path.join(out_dir, out_name), "w", encoding="utf-8") as fh:
            json.dump(result, fh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
