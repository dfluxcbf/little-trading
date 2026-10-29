"""One module per lab service (an alternative or an improvement found in the research).

A service module defines `run(ctx)`. It may import only the lab, the cost model, the restricted-issuer list and the
pure feature code (see `little_trading.lab.service.isolation`); the lab refuses to register one that can reach the
trading engine, server, store or web code.
"""
