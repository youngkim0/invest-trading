"""Hyperliquid whale position tracker.

Tracks positions of top-performing traders on Hyperliquid DEX.
Wallet addresses sourced from Hyperliquid leaderboard (top 30 by volume/PnL).
Positions are public on-chain — no API key required.
"""

import asyncio
from datetime import datetime, timezone
from typing import Any

import httpx
from loguru import logger

# Hyperliquid API endpoint
HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"
HYPERLIQUID_LEADERBOARD_URL = "https://stats-data.hyperliquid.xyz/Mainnet/leaderboard"

# Top 40 whale wallets from Hyperliquid leaderboard (sorted by 30D ROI%, Mar 2026)
# Criteria: >5% monthly ROI, >$500k account, positive alltime PnL
# These are the best *directional* traders right now, not just biggest accounts
WHALE_WALLETS = [
    "0x5bcb8556d5e8d7fee989cf06045a4275e5d9b800",  # +52554%/mo, $2.5M acct
    "0xd3cb1823da2ff584dec3f49ef6a3eea51471e5bc",  # +1284%/mo, $12.2M acct
    "0x5b28335f6d5cd5a1ac3b66eb2edd2b917e7aa4a4",  # +1114%/mo, $1M acct
    "0xeb2c29a8917422b3ff466f74539b206318c40118",  # +567%/mo, $643k acct
    "0x1c81fb7100276aef766092e16cfcf61097e97a5d",  # +243%/mo, $651k acct
    "0x3b11267dfc4b9ebe8427e8f557056b4b6ce98112",  # +203%/mo, $552k acct
    "0x049bdc370620beab340b01072fa580fd57745e7d",  # ♤♤♤ — +200%/mo, $6.7M acct
    "0x90dbb196cacf24bd212b377a0dd62eaad5e8151b",  # +163%/mo, $2.7M acct
    "0x198d12ecc9b499adf122eb62159dae75c6ca8002",  # +159%/mo, $630k acct
    "0x9cd0a696c7cbb9d44de99268194cb08e5684e5fe",  # +153%/mo, $4.1M acct
    "0x20b7807939c774a152b396beb704acac68fa10fe",  # +150%/mo, $1.6M acct
    "0xdd7a372377fc633f74ab6e20963803d52f448830",  # +128%/mo, $553k acct
    "0x8c6d04086236a7c1670a967fc393ff58ca1d7ce9",  # +125%/mo, $632k acct
    "0xed48b856556a69c7c40229c9c4c829b909257c9b",  # +123%/mo, $1.5M acct
    "0xedf2b293d5b358f17330c8412e0be36feaa8fc0b",  # +118%/mo, $2.8M acct
    "0x6b08bbb2daf57d390538d73542f50d5f735ca420",  # +103%/mo, $1.1M acct
    "0xa98143608a6846453c88f29506d64a8b85532e65",  # +101%/mo, $4.1M acct
    "0xfe40fa3decfc6dc555f4ad7736d0dd65ccbd4743",  # +100%/mo, $2M acct
    "0xc6758a779bccee1ef0190dbe8292fdf44076795d",  # +96%/mo, $1.9M acct
    "0x795cfd1b03eafc11c4ec958b8a94cfc9aa64a242",  # +91%/mo, $2.4M acct
    "0x6a02aedceac5a6813d960e4dae1910d9c458e77c",  # +91%/mo, $570k acct
    "0xe9bf81b432e5bf34995afae08747c530c8406c4d",  # +88%/mo, $797k acct
    "0xfeedc4f156fd25c3fe9ff6402dbf3b99ebed3195",  # +85%/mo, $911k acct
    "0x2fc3195efbf91ad90854bc3c02fe739895c23460",  # +84%/mo, $3M acct
    "0xa11f83cbb07fd0327415bd424b986132e009f643",  # +79%/mo, $1.1M acct
    "0x24a44aef48aeb27c7708dabfccda14b41fbf0ae1",  # +76%/mo, $728k acct
    "0x1def62ffe2a62d65f649991d7f2199e046a30f72",  # +75%/mo, $999k acct
    "0x27388d079cb5cf1ad50f60ab4c260356449bf92b",  # +74%/mo, $546k acct
    "0x547c8d938b98cd17ab7c653a0d98cca80eabb876",  # +73%/mo, $2.6M acct
    "0x18cd4597e06b7fe0a8cd33dda499121b3a145a8b",  # +72%/mo, $2.4M acct
    "0x420a4ed7b6bb361da586868adec2f2bb9ab75e66",  # +70%/mo, $837k acct
    "0xbd9c944dcfb31cd24c81ebf1c974d950f44e42b8",  # guy.neet — +70%/mo, $652k acct
    "0xab961d7c42bbcd454a54b342bd191a8f090219e6",  # +69%/mo, $5M acct
    "0xb58e1eb8256689e47f589ba6f6a28657d997bf2f",  # +69%/mo, $1.5M acct
    "0xce5667f7194c4d9320acd28f1ee8b3d2adbdb27e",  # +64%/mo, $789k acct
    "0x75ec3ba266176e733f1b1fdaa15052f5eff724b8",  # +64%/mo, $1.3M acct
    "0xd210bd2ed3fafe7a4de5b079392823c81b0dc56a",  # +63%/mo, $1.4M acct
    "0xb42943e70cb3e27ce45c8c6008b623f292b5dd04",  # +62%/mo, $872k acct
    "0x888e000c78b8f1aada5b3c99f880794907b76d77",  # +60%/mo, $3.1M acct
    "0xe091330c53b971ed4e886f902f18e8544a8d6705",  # +58%/mo, $1.1M acct
]

# Map Hyperliquid coin names to Binance symbol format
COIN_TO_SYMBOL = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "XRP": "XRPUSDT",
    "DOGE": "DOGEUSDT",
    "AVAX": "AVAXUSDT",
}

# Only track coins we actually trade
TRACKED_COINS = set(COIN_TO_SYMBOL.keys())


class HyperliquidWhaleTracker:
    """Track positions of top Hyperliquid traders for smart money signals."""

    # Leaderboard refresh config
    REFRESH_DAY = 0  # Monday (0=Mon, 6=Sun)
    REFRESH_HOUR = 0  # 00:00 UTC
    MIN_MONTHLY_ROI = 0.05  # >5% monthly ROI
    MIN_ACCOUNT_VALUE = 500_000  # >$500k account
    MAX_WALLETS = 40

    def __init__(self, wallets: list[str] | None = None):
        self.wallets = wallets or list(WHALE_WALLETS)  # mutable copy
        self.client = httpx.AsyncClient(timeout=15.0)
        self._cache: dict[str, Any] = {}
        self._cache_time: datetime | None = None
        self._cache_ttl_seconds = 120  # Cache whale data for 2 minutes
        self._last_leaderboard_refresh: datetime | None = None

    async def close(self):
        await self.client.aclose()

    async def maybe_refresh_wallets(self):
        """Refresh wallet list from leaderboard weekly (Monday 00:00 UTC).

        Also refreshes on first call (startup).
        """
        now = datetime.now(timezone.utc)

        # First call: always refresh
        if self._last_leaderboard_refresh is None:
            await self._refresh_from_leaderboard()
            return

        # Check if it's Monday and we haven't refreshed today
        if (now.weekday() == self.REFRESH_DAY
                and now.hour >= self.REFRESH_HOUR
                and (now - self._last_leaderboard_refresh).total_seconds() > 86400):
            await self._refresh_from_leaderboard()

    async def _refresh_from_leaderboard(self):
        """Fetch leaderboard and update wallet list sorted by 30D ROI%."""
        try:
            response = await self.client.get(HYPERLIQUID_LEADERBOARD_URL, timeout=30.0)
            response.raise_for_status()
            data = response.json()
            rows = data.get("leaderboardRows", [])

            candidates = []
            for r in rows:
                addr = r.get("ethAddress", "")
                acct_val = float(r.get("accountValue", 0))
                perfs = {p[0]: p[1] for p in r.get("windowPerformances", [])}

                month_perf = perfs.get("month", {})
                month_roi = float(month_perf.get("roi", 0))
                alltime_pnl = float(perfs.get("allTime", {}).get("pnl", 0))

                if (month_roi > self.MIN_MONTHLY_ROI
                        and acct_val > self.MIN_ACCOUNT_VALUE
                        and alltime_pnl > 0):
                    candidates.append((addr, month_roi, acct_val))

            # Sort by 30D ROI% descending, take top N
            candidates.sort(key=lambda x: -x[1])
            new_wallets = [addr for addr, _, _ in candidates[:self.MAX_WALLETS]]

            if new_wallets:
                old_set = set(self.wallets)
                new_set = set(new_wallets)
                added = len(new_set - old_set)
                removed = len(old_set - new_set)
                self.wallets = new_wallets
                logger.info(
                    f"🐋 Whale leaderboard refreshed: {len(new_wallets)} wallets "
                    f"(+{added} new, -{removed} dropped, "
                    f"top ROI: {candidates[0][1]*100:.0f}%/mo)"
                )
            else:
                logger.warning("🐋 Leaderboard refresh returned 0 candidates, keeping existing list")

            self._last_leaderboard_refresh = datetime.now(timezone.utc)

        except Exception as e:
            logger.warning(f"🐋 Leaderboard refresh failed (keeping existing list): {e}")
            self._last_leaderboard_refresh = datetime.now(timezone.utc)

    async def _get_wallet_positions(self, address: str) -> list[dict]:
        """Get open positions for a single wallet."""
        try:
            response = await self.client.post(
                HYPERLIQUID_INFO_URL,
                json={"type": "clearinghouseState", "user": address},
            )
            response.raise_for_status()
            data = response.json()
            positions = []
            for p in data.get("assetPositions", []):
                pos = p.get("position", {})
                coin = pos.get("coin", "")
                if coin not in TRACKED_COINS:
                    continue
                size = float(pos.get("szi", 0))
                if size == 0:
                    continue
                positions.append({
                    "coin": coin,
                    "side": "long" if size > 0 else "short",
                    "size": abs(size),
                    "entry_price": float(pos.get("entryPx", 0)),
                    "unrealized_pnl": float(pos.get("unrealizedPnl", 0)),
                    "leverage": pos.get("leverage", {}).get("value", 1),
                })
            return positions
        except Exception as e:
            logger.debug(f"Failed to fetch HL wallet {address[:10]}...: {e}")
            return []

    async def get_whale_consensus(self) -> dict[str, dict]:
        """Get aggregated whale positioning for all tracked coins.

        Returns:
            {symbol: {
                long_count: int, short_count: int, total: int,
                consensus: float (-1.0 to +1.0),
                long_value: float, short_value: float,
            }}
        """
        # Auto-refresh wallet list (weekly on Monday, or on first call)
        await self.maybe_refresh_wallets()

        # Check cache
        now = datetime.now(timezone.utc)
        if (self._cache_time and self._cache
                and (now - self._cache_time).total_seconds() < self._cache_ttl_seconds):
            return self._cache

        # Query all wallets with small delays to be respectful
        all_positions = []
        for i, wallet in enumerate(self.wallets):
            positions = await self._get_wallet_positions(wallet)
            all_positions.append((wallet, positions))
            if i < len(self.wallets) - 1:
                await asyncio.sleep(0.1)  # 100ms between requests

        # Aggregate by coin
        consensus = {}
        for symbol_name, binance_symbol in COIN_TO_SYMBOL.items():
            long_count = 0
            short_count = 0
            long_value = 0.0
            short_value = 0.0

            for wallet, positions in all_positions:
                for pos in positions:
                    if pos["coin"] == symbol_name:
                        notional = pos["size"] * pos["entry_price"]
                        if pos["side"] == "long":
                            long_count += 1
                            long_value += notional
                        else:
                            short_count += 1
                            short_value += notional

            total = long_count + short_count
            if total > 0:
                # Consensus: +1.0 = all whales long, -1.0 = all whales short
                score = (long_count - short_count) / total
            else:
                score = 0.0

            consensus[binance_symbol] = {
                "long_count": long_count,
                "short_count": short_count,
                "total": total,
                "consensus": score,
                "long_value": long_value,
                "short_value": short_value,
            }

        # Update cache
        self._cache = consensus
        self._cache_time = now

        active = sum(1 for c in consensus.values() if c["total"] > 0)
        logger.info(
            f"🐋 Whale consensus updated: {active} coins with positions "
            f"({sum(w[1] != [] for w in all_positions)}/{len(self.wallets)} wallets responded)"
        )

        return consensus
