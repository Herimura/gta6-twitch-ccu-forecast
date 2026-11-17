# Comparable-Curve Forecaster

[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Herimura/gta6-twitch-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

An entry for make.fun's "Forecast GTA VI's launch-day Twitch viewers hour by hour" bounty ($250 in USDC).

`main.py` (Python, standard library only, deterministic) forecasts the hour-by-hour
Twitch CCU curve of a game launch from the case's own comparable launches: the
hour-1 print is projected along similarity-weighted empirical quantile curves of
past launches' log-ratio trajectories, blended toward a parametric curve fitted to
the weighted median (log-hour polynomial plus release-wave bumps), with the spread
calibrated around the median.
