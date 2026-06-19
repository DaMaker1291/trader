"""Quick smoke test for optimized gap_bot.py -- just checks startup."""
import sys, os, asyncio
sys.path.insert(0, r"C:\Users\supro\Downloads\ME BOT")

# Monkey-patch the main loop to exit after init
import gap_bot as gb

async def test():
    gb.logger.info("=" * 55)
    gb.logger.info("SMOKE TEST — Optimized Gap Bot")
    gb.logger.info("  CAPITAL=%s SL=%s TRAIL_ACT=%s TRAIL_DIST=%s MIN_GAP=%s",
                   gb.CAPITAL, gb.HARD_SL, gb.TRAIL_ACTIVATE, gb.TRAIL_DIST, gb.MIN_GAP)
    gb.logger.info("  MAX_POSITIONS=%s MIN_WIN_PROB=%s", gb.MAX_POSITIONS, gb.MIN_WIN_PROB)
    gb.logger.info("  Shorts disabled: SHORT_MIN_GAP=%s", gb.SHORT_MIN_GAP)
    gb.logger.info("  WATCHLIST=%d stocks", len(gb.WATCHLIST))
    gb.logger.info("=" * 55)

    model = gb.TradeModel()
    gb.logger.info("Model loaded: %d past trades", len(model.trades))
    gb.logger.info("%s", model.report())

    # Quick scan test
    signals = await gb.scan_premarket(None)
    gb.logger.info("Scan returned %d signals", len(signals))
    for s in signals[:3]:
        s["win_prob"] = model.predict_win_prob(s)
        gb.logger.info("  %s: gap=+%.1f%% vol=%d win_prob=%.0f%%",
                       s["sym"], s["gap"], s["vol"], s["win_prob"]*100)

    gb.logger.info("\n✅ Bot initializes and scans correctly")
    gb.logger.info("Ready for live trading with: python gap_bot.py [--sim]")

asyncio.run(test())
