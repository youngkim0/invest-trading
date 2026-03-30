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

# Top 20 whale wallets from Hyperliquid leaderboard (sorted by monthly PnL, Mar 2026)
# Filtered to profitable traders with >$1M account value
WHALE_WALLETS = [
    "0x162cc7c861ebd0c06b3d72319201150482518185",  # ABC — $41M alltime PnL
    "0x87f9cd15f5050a9283b8896300f7c8cf69ece2cf",  # $53M alltime, $81M acct
    "0xecb63caa47c7c4e77f60f1ce858cf28dc2b82b00",  # $203M alltime, $76M acct
    "0xfc667adba8d4837586078f4fdcdc29804337ca06",  # $18M alltime, $79M acct
    "0xff4cd3826ecee12acd4329aada4a2d3419fc463c",  # $26M alltime, $55M acct
    "0x023a3d058020fb76cca98f01b3c48c8938a22355",  # Auros — $64M alltime
    "0xefd3ab65915e35105caa462442c9ecc1346728df",  # 2 frères — $7.5M alltime
    "0xdcac85ecae7148886029c20e661d848a4de99ce2",  # $22M alltime
    "0x31dea2516beee92135b96f464eeec3cf292a13f2",  # $16.5M alltime
    "0x57dd78cd36e76e2011e8f6dc25cabbaba994494b",  # $17.8M acct
    "0xe357fa9fecb084f0303ff341b0bc55c89f2bb5ce",  # $15.5M acct
    "0xc926ddba8b7617dbc65712f20cf8e1b58b8598d3",  # $13.3M acct, +16.5% month
    "0x85ecf584f25db6f146718b86d493e33c5af72052",  # $13.5M acct, +9.3% month
    "0x3bcae23e8c380dab4732e9a159c0456f12d866f3",  # $12.3M acct, $11.9M alltime
    "0x61ceef212ff4a86933c69fb6aca2fe35d8f2a62b",  # $12.8M acct, +35.3% month
    "0x7839e2f2c375dd2935193f2736167514efff9916",  # $8.5M acct, +19.2% month
    "0x399965e15d4e61ec3529cc98b7f7ebb93b733336",  # $8.6M acct, +25.7% month
    "0xeeb56331b6a250fe2dbc123f08bdb87aa9840464",  # $8.2M acct
    "0xe4c6ae25959d7fc66cf2dd5965fb78c5e09c4048",  # $14M alltime PnL
    "0xdf9ea6ec3b7109935ccb4fb267e15ac1fb077ab1",  # $9.4M alltime PnL
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

    def __init__(self, wallets: list[str] | None = None):
        self.wallets = wallets or WHALE_WALLETS
        self.client = httpx.AsyncClient(timeout=15.0)
        self._cache: dict[str, Any] = {}
        self._cache_time: datetime | None = None
        self._cache_ttl_seconds = 120  # Cache whale data for 2 minutes

    async def close(self):
        await self.client.aclose()

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
