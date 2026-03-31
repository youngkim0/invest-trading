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

# Top 40 whale wallets from Hyperliquid leaderboard (sorted by allTime PnL, Mar 2026)
# Filtered to profitable traders with >$500k account value
WHALE_WALLETS = [
    "0xecb63caa47c7c4e77f60f1ce858cf28dc2b82b00",  # $82M acct, $203M alltime
    "0x5b5d51203a0f9079f8aeb098a6523a13f298c060",  # $36M acct, $173M alltime
    "0x7fdafde5cfb5465924316eced2d3715494c517d1",  # BobbyBigSize — $31M acct, $162M alltime
    "0xfae95f601f3a25ace60d19dbb929f2a5c57e3571",  # thank you jefef — $7M acct, $150M alltime
    "0xdfc24b077bc1425ad1dea75bcb6f8158e10df303",  # $463M acct, $138M alltime
    "0x20c2d95a3dfdca9e9ad12794d5fa6fad99da44f5",  # $3.5M acct, $122M alltime
    "0xb83de012dba672c76a7dbbbf3e459cb59d7d6e36",  # $48M acct, $119M alltime
    "0x880ac484a1743862989a441d6d867238c7aa311c",  # x35767 — $28M acct, $114M alltime
    "0xa312114b5795dff9b8db50474dd57701aa78ad1e",  # $12M acct, $93M alltime
    "0x716bd8d3337972db99995dda5c4b34d954a61d95",  # $48M acct, $89M alltime
    "0xd47587702a91731dc1089b5db0932cf820151a91",  # $44M acct, $86M alltime
    "0x2e3d94f0562703b25c83308a05046ddaf9a8dd14",  # $1M acct, $85M alltime
    "0x45d26f28196d226497130c4bac709d808fed4029",  # $25M acct, $82M alltime
    "0xbdfa4f4492dd7b7cf211209c4791af8d52bf5c50",  # $38M acct, $71M alltime
    "0x023a3d058020fb76cca98f01b3c48c8938a22355",  # Auros — $30M acct, $63M alltime
    "0x8e096995c3e4a3f0bc5b3ea1cba94de2aa4d70c9",  # $7.8M acct, $60M alltime
    "0x5d2f4460ac3514ada79f5d9838916e508ab39bb7",  # $5M acct, $60M alltime
    "0x35d1151ef1aab579cbb3109e69fa82f94ff5acb1",  # $19M acct, $59M alltime
    "0x493db0ed7514c975e9abcc110bd40c473b6763e3",  # $49M acct, $59M alltime
    "0x8af700ba841f30e0a3fcb0ee4c4a9d223e1efa05",  # $17M acct, $58M alltime
    "0xcfdb74a8c080bb7b4360ed6fe21f895c653efff4",  # $34M acct, $56M alltime
    "0x4e14fc11f58b64740e66e4b1aa188a4b007c0eab",  # $55M acct, $53M alltime
    "0x87f9cd15f5050a9283b8896300f7c8cf69ece2cf",  # $80M acct, $52M alltime
    "0x0d446c3372a9ba9cddef0eef7a1afab6dc0e8c0b",  # $52M acct, $52M alltime
    "0x13c50dcdee4bbcba71baf578b345cdd35c7928be",  # $35M acct, $47M alltime
    "0x010461c14e146ac35fe42271bdc1134ee31c703a",  # $134M acct, $46M alltime
    "0x939f95036d2e7b6d7419ec072bf9d967352204d2",  # $39M acct, $46M alltime
    "0x7dacca323e44f168494c779bb5e7483c468ef410",  # $31M acct, $46M alltime
    "0xcac19662ec88d23fa1c81ac0e8570b0cf2ff26b3",  # $26M acct, $46M alltime
    "0x03b9a189e2480d1e4c3007080b29f362282130fa",  # $38M acct, $43M alltime
    "0x162cc7c861ebd0c06b3d72319201150482518185",  # ABC — $40M acct, $41M alltime
    "0x856c35038594767646266bc7fd68dc26480e910d",  # $33M acct, $38M alltime
    "0xa87a233e8a7d8951ff790a2e39738086cb5f71b7",  # $14M acct, $38M alltime
    "0x31ca8395cf837de08b24da3f660e77761dfb974b",  # $135M acct, $37M alltime
    "0x418aa6bf98a2b2bc93779f810330d88cde488888",  # $10M acct, $37M alltime
    "0xc59498175d6d317642aeb97f895a7ce1aa992191",  # $4.5M acct, $34M alltime
    "0x82d8dc80190e6bc1d92b048f9fc7e85e5e1e32ff",  # $22M acct, $33M alltime
    "0x1419e75330c71ce463102e6a1eb62fe80b412d5f",  # $29M acct, $32M alltime
    "0xd4c1f7e8d876c4749228d515473d36f919583d1d",  # $8.7M acct, $30M alltime
    "0xb0a55f13d22f66e6d495ac98113841b2326e9540",  # $30M acct, $30M alltime
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
