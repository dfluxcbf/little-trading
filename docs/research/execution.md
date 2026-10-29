# Orders and execution (F079-F087)

Today: the simulated portfolio fills market orders instantly at one price, whole shares only, and real order
execution is a non-goal. These items keep that non-goal: anything that touches a broker is about paper
accounts unless a later design decision says otherwise.

| ID | Feature | Mode | What and why | Priority | Effort |
|---|---|---|---|---|---|
| F079 | Broker paper-trading adapter | Auto | Send orders to a broker's paper account (Alpaca's paper API) for realistic fills, queues and rejections, still with no real money. Real execution stays a separate, explicit decision. | P2 | M |
| F080 | Order types | Auto | Limit, stop, stop-limit, bracket and one-cancels-other orders, with time-in-force. | P2 | M |
| F081 | Order lifecycle | Auto | Track each order through pending, partial fill, filled, cancelled and rejected, instead of assuming immediate fills. | P2 | M |
| F082 | Execution timing | Auto | Avoid the first and last minutes of the session, and slice larger orders over time (TWAP or VWAP). | P3 | M |
| F083 | Fractional shares | Auto | Size in fractions so small accounts can hold expensive stocks at the intended weight. | P3 | S |
| F084 | Reconciliation | Auto | Compare local positions and cash with the broker's every few minutes and stop on a mismatch. | P2 | S |
| F085 | Several portfolios | Both | Separate portfolios per strategy or horizon, each with its own state and limits. | P2 | M |
| F086 | Pre-trade compliance | Auto | Pattern-day-trader limits on margin accounts under $25,000, wash-trade checks and settled-cash rules, checked before each order. | P2 | M |
| F087 | Event-driven live loop | Auto | React to new bars, price moves and news as they arrive instead of sleeping a fixed interval. | P2 | M |
