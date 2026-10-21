# Protocol
Hypothesis: Pairs of consumer-staples stocks that are cointegrated over a
2-year formation window remain mean-reverting over the following 6 months.
Universe: 14 US consumer staples stocks (see data.py). Daily closes, 2012-2024.
Method: walk-forward. 504-day formation window, 126-day trading window.
Timing: signal at close of day t -> position held during day t+1.
Costs: 5 bps per unit traded + 0.5% annual borrow on short leg.
Train period (design/tuning allowed): trades before 2019-01-01.
Test period (look ONCE at the end): trades from 2019-01-01.
Success: net Sharpe > 0.5 on the test period.
Known limits: survivorship bias (universe = today's survivors),
adjusted close prices, no bid/ask data.