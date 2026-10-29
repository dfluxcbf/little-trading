# Interface, reporting and accounting (F096-F100)

Today: a CLI that prints trades and one-line summaries. Nothing is shown over time, and no person is in the
loop.

| ID | Feature | Mode | What and why | Priority | Effort |
|---|---|---|---|---|---|
| F096 | Web dashboard | Manual | Portfolio, positions, equity curve against benchmark, current signals with rationale, and the news feed per symbol. | P2 | L |
| F097 | Approve-before-trade mode | Manual | A semi-automatic mode where the app proposes trades and the user approves or rejects each; the answers also feed Jeff (F045). | P2 | M |
| F098 | Performance attribution | Both | Daily and weekly summaries, with P&L split by what drove each trade (technical, news, Jeff) and by horizon. | P2 | M |
| F099 | Tax lots | Both | FIFO lot accounting, short- and long-term realised gains, wash-sale loss tracking and a yearly export. | P3 | M |
| F100 | What-if planner | Manual | Enter a hypothetical trade and see the simulated distribution of outcomes and the risk it adds to the portfolio. | P3 | M |
