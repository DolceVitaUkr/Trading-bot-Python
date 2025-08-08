# modules/top_pairs.py

import requests
import logging
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Literal

import config

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class PairManager:
    """
    Fetch and cache active USDT spot pairs from Bybit.

    - Primary source: Bybit Spot symbols API (live data).
    - Cache: JSON file with TTL (1h by default) to avoid hammering the API.
    - Fallbacks: user-provided file, then a small hardcoded list.
    - Output format: return CCXT-style (e.g., 'BTC/USDT') by default.

    Configurable via:
      - config.PAIR_CACHE_FILE (default: "pair_cache.json")
      - config.PAIR_CACHE_TTL_HOURS (default: 1)
      - config.PAIR_FALLBACK_FILE (default: "fallback_pairs.json")
    """

    API_ENDPOINT = "https://api.bybit.com/spot/v1/symbols"

    def __init__(self):
        self.cache_file: str = getattr(config, "PAIR_CACHE_FILE", "pair_cache.json")
        self.cache_ttl: timedelta = timedelta(
            hours=float(getattr(config, "PAIR_CACHE_TTL_HOURS", 1))
        )
        self.fallback_file: str = getattr(config, "PAIR_FALLBACK_FILE", "fallback_pairs.json")
        self.headers = {
            "User-Agent": "AI-Trading-Bot/1.0 (+https://github.com/DolceVitaUkr/Trading-bot-Python)",
            "Accept": "application/json",
        }

    def get_active_pairs(
        self, fmt: Literal["ccxt", "bybit"] = "ccxt"
    ) -> List[str]:
        """Return a list of currently tradeable USDT spot pairs."""
        try:
            pairs_bybit = self._fetch_api_pairs()
            if pairs_bybit:
                self._update_cache(pairs_bybit)
                return self._format_pairs(pairs_bybit, fmt)
        except Exception as e:
            logger.warning(f"Pair API fetch failed: {e}")

        cached = self._get_cached_pairs()
        if cached:
            logger.info(f"Loaded {len(cached)} pairs from cache")
            return self._format_pairs(cached, fmt)

        file_fb = self._get_file_fallback()
        if file_fb:
            logger.info(f"Loaded {len(file_fb)} pairs from fallback file")
            return self._format_pairs(file_fb, fmt)

        hard = self._get_hardcoded_fallback()
        logger.warning(f"Using hardcoded fallback pairs ({len(hard)} total)")
        return self._format_pairs(hard, fmt)

    def _fetch_api_pairs(self) -> Optional[List[str]]:
        """Hit Bybit API, validate, and extract list of USDT pairs (BYBIT format)."""
        resp = requests.get(self.API_ENDPOINT, headers=self.headers, timeout=(3.05, 10))
        resp.raise_for_status()
        data = resp.json()

        if not self._validate_api_response(data):
            raise ValueError(f"Invalid API response structure: {data.keys()}")

        symbols: List[Dict] = data["result"]
        return self._process_pairs(symbols)

    @staticmethod
    def _validate_api_response(data: Dict) -> bool:
        return data is not None and data.get("ret_code") == 0 and isinstance(data.get("result"), list)

    @staticmethod
    def _process_pairs(symbols: List[Dict]) -> List[str]:
        return sorted({
            s.get("name")
            for s in symbols
            if s.get("quoteCurrency") == "USDT"
            and s.get("status") == "Trading"
            and isinstance(s.get("name"), str)
        })

    def _update_cache(self, pairs_bybit: List[str]) -> None:
        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "pairs": list(pairs_bybit),
        }
        try:
            with open(self.cache_file, "w") as f:
                json.dump(payload, f)
        except Exception as e:
            logger.error(f"Failed to update pair cache: {e}")

    def _get_cached_pairs(self) -> Optional[List[str]]:
        if not os.path.exists(self.cache_file):
            return None
        try:
            with open(self.cache_file, "r") as f:
                data = json.load(f)
            ts = datetime.fromisoformat(data.get("timestamp", datetime.utcnow().isoformat()))
            if datetime.utcnow() - ts < self.cache_ttl:
                return list(data.get("pairs", []))
        except Exception as e:
            logger.error(f"Error reading pair cache: {e}")
        return None

    def _get_file_fallback(self) -> Optional[List[str]]:
        if not os.path.exists(self.fallback_file):
            return None
        try:
            with open(self.fallback_file, "r") as f:
                data = json.load(f)
            return [self._to_bybit(p) for p in data if isinstance(p, str)] or None
        except Exception as e:
            logger.error(f"Error reading fallback file: {e}")
            return None

    @staticmethod
    def _get_hardcoded_fallback() -> List[str]:
        return [
            "1INCHUSDT", "1SOLUSDT", "3PUSDT", "5IREUSDT", "A8USDT", "AARKUSDT", "AAVEUSDT", "ACAUSDT",
            "ACHUSDT", "ACSUSDT", "ADAUSDT", "AEGUSDT", "AEROUSDT", "AEVOUSDT", "AFCUSDT", "AFGUSDT",
            "AGIUSDT", "AGLAUSDT", "AGLDUSDT", "AI16ZUSDT", "AIOZUSDT", "AIXBTUSDT", "AKIUSDT", "ALCHUSDT",
            "ALGOUSDT", "ALTUSDT", "ANIMEUSDT", "ANKRUSDT", "APEUSDT", "APEXUSDT", "APPUSDT", "APRSUSDT",
            "APTRUSDT", "APTUSDT", "ARBUSDT", "ARKMUSDT", "ARTYUSDT", "ARUSDT", "ATHUSDT", "ATOMUSDT",
            "AURORAUSDT", "AVAILUSDT", "AVAUSDT", "AVAXUSDT", "AVLUSDT", "AXLUSDT", "AXSUSDT", "B3USDT",
            "BABYDOGEUSDT", "BANUSDT", "BATUSDT", "BBLUSDT", "BBQUSDT", "BBSOLUSDT", "BBUSDT", "BCHUSDT",
            "BCUTUSDT", "BEAMUSDT", "BELUSDT", "BERAUSDT", "BICOUSDT", "BLASTUSDT", "BLURUSDT", "BMTUSDT",
            "BNBUSDT", "BOBAUSDT", "BOBUSDT", "BOMEUSDT", "BONKUSDT", "BONUSUSDT", "BRAWLUSDT", "BRETTUSDT",
            "BRUSDT", "BTC3LUSDT", "BTC3SUSDT", "BTCUSDT", "BTTUSDT", "C98USDT", "CAKEUSDT", "CAPSUSDT",
            "CARVUSDT", "CATBNBUSDT", "CATIUSDT", "CATSUSDT", "CBKUSDT", "CELOUSDT", "CELUSDT", "CGPTUSDT",
            "CHILLGUYUSDT", "CHRPUSDT", "CHZUSDT", "CITYUSDT", "CLOUDUSDT", "CMETHUSDT", "COMPUSDT",
            "COOKIEUSDT", "COOKUSDT", "COQUSDT", "COREUSDT", "COTUSDT", "CPOOLUSDT", "CRVUSDT", "CSPRUSDT",
            "CTAUSDT", "CTCUSDT", "CTTUSDT", "CYBERUSDT", "DAIUSDT", "DBRUSDT", "DEEPUSDT", "DEFIUSDT",
            "DEGENUSDT", "DGBUSDT", "DIAMUSDT", "DICEUSDT", "DLCUSDT", "DMAILUSDT", "DOGEUSDT", "DOGSUSDT",
            "DOMEUSDT", "DOP1USDT", "DOTUSDT", "DRIFTUSDT", "DSRUNUSDT", "DUELUSDT", "DYDXUSDT", "DYMUSDT",
            "ECOXUSDT", "EGLDUSDT", "EGOUSDT", "EGP1USDT", "EIGENUSDT", "ELDAUSDT", "ELIXUSDT", "ELXUSDT",
            "ENAUSDT", "ENJUSDT", "ENSUSDT", "EOSUSDT", "ERTHAUSDT", "ESEUSDT", "ETCUSDT", "ETH3LUSDT",
            "ETH3SUSDT", "ETHFIUSDT", "ETHUSDT", "ETHWUSDT", "EVERUSDT", "EXVGUSDT", "FAMEUSDT", "FARUSDT",
            "FETUSDT", "FIDAUSDT", "FILUSDT", "FIREUSDT", "FITFIUSDT", "FLIPUSDT", "FLOCKUSDT", "FLOKIUSDT",
            "FLOWUSDT", "FLRUSDT", "FLTUSDT", "FLUIDUSDT", "FMBUSDT", "FMCUSDT", "FORTUSDT", "FOXYUSDT",
            "FTTUSDT", "FUELUSDT", "FUSDT", "FXSUSDT", "G3USDT", "G7USDT", "GALAUSDT", "GALAXISUSDT",
            "GALFTUSDT", "GAMEUSDT", "GENEUSDT", "GLMRUSDT", "GMRXUSDT", "GMTUSDT", "GMXUSDT", "GOATUSDT",
            "GODSUSDT", "GPSUSDT", "GRAPEUSDT", "GRASSUSDT", "GRTUSDT", "GSTSUSDT", "GSTUSDT", "GSWIFTUSDT",
            "GTAIUSDT", "GUMMYUSDT", "GUSDT", "HATUSDT", "HBARUSDT", "HFTUSDT", "HLGUSDT", "HMSTRUSDT",
            "HNTUSDT", "HOOKUSDT", "HOTUSDT", "HPOS10IUSDT", "HTXUSDT", "HVHUSDT", "ICPUSDT", "ICXUSDT",
            "IDUSDT", "IMXUSDT", "INJUSDT", "INSPUSDT", "INTERUSDT", "IOUSDT", "IPUSDT", "IZIUSDT",
            "JASMYUSDT", "JEFFUSDT", "JSTUSDT", "JTOUSDT", "JUPUSDT", "JUSDT", "JUVUSDT", "KAIAUSDT",
            "KARATEUSDT", "KASTAUSDT", "KASUSDT", "KAVAUSDT", "KCALUSDT", "KCSUSDT", "KDAUSDT", "KMNOUSDT",
            "KONUSDT", "KROUSDT", "KSMUSDT", "KUBUSDT", "L3USDT", "LADYSUSDT", "LAIUSDT", "LAVAUSDT",
            "LAYERUSDT", "LDOUSDT", "LEVERUSDT", "LFTUSDT", "LGXUSDT", "LINKUSDT", "LISUSDT", "LLUSDT",
            "LMWRUSDT", "LOOKSUSDT", "LRCUSDT", "LTCUSDT", "LUCEUSDT", "LUNAIUSDT", "LUNAUSDT", "LUNCUSDT",
            "MAGICUSDT", "MAJORUSDT", "MAKUSDT", "MANAUSDT", "MANTAUSDT", "MASAUSDT", "MASKUSDT",
            "MAVIAUSDT", "MBOXUSDT", "MBSUSDT", "MBXUSDT", "MCRTUSDT", "MDAOUSDT", "MEEUSDT", "MEMEFIUSDT",
            "MEMEUSDT", "MERLUSDT", "METHUSDT", "MEUSDT", "MEWUSDT", "MINAUSDT", "MIXUSDT", "MKRUSDT",
            "MLKUSDT", "MNRYUSDT", "MNTUSDT", "MOCAUSDT", "MODEUSDT", "MOGUSDT", "MOJOUSDT", "MONUSDT",
            "MORPHOUSDT", "MOVEUSDT", "MOVRUSDT", "MOZUSDT", "MPLXUSDT", "MSTARUSDT", "MVLUSDT", "MVUSDT",
            "MXMUSDT", "MXUSDT", "MYRIAUSDT", "MYROUSDT", "N3USDT", "NAKAUSDT", "NAVXUSDT", "NEARUSDT",
            "NEIROCTOUSDT", "NEIROUSDT", "NEONUSDT", "NESSUSDT", "NEXOUSDT", "NFTUSDT", "NGLUSDT",
            "NIBIUSDT", "NLKUSDT", "NOTUSDT", "NRNUSDT", "NSUSDT", "NUTSUSDT", "NYANUSDT", "NYMUSDT",
            "OASUSDT", "OBTUSDT", "OBXUSDT", "ODOSUSDT", "OIKUSDT", "OLUSDT", "OMGUSDT", "OMNIUSDT",
            "OMUSDT", "ONDOUSDT", "ONEUSDT", "OPUSDT", "ORDERUSDT", "ORDIUSDT", "ORTUSDT", "PAALUSDT",
            "PBUXUSDT", "PELLUSDT", "PENDLEUSDT", "PENGUUSDT", "PEOPLEUSDT", "PEPEUSDT", "PERPUSDT",
            "PINEYEUSDT", "PIPUSDT", "PIRATEUSDT", "PIXFIUSDT", "PLTUSDT", "PLUMEUSDT", "PNUTUSDT",
            "POKTUSDT", "POLUSDT", "PONKEUSDT", "POPCATUSDT", "PORT3USDT", "PORTALUSDT", "PPTUSDT",
            "PRCLUSDT", "PRIMEUSDT", "PSGUSDT", "PSTAKEUSDT", "PTCUSDT", "PTUUSDT", "PUFFERUSDT",
            "PUFFUSDT", "PUMLXUSDT", "PURSEUSDT", "PYTHUSDT", "PYUSDUSDT", "QNTUSDT", "QORPOUSDT",
            "QTUMUSDT", "RACAUSDT", "RAINUSDT", "RATSUSDT", "RDNTUSDT", "REALUSDT", "REDUSDT",
            "RENDERUSDT", "RENUSDT", "ROAMUSDT", "RONDUSDT", "ROOTUSDT", "ROSEUSDT", "RPKUSDT",
            "RSS3USDT", "RUNEUSDT", "RVNUSDT", "SAFEUSDT", "SAILUSDT", "SALDUSDT", "SANDUSDT",
            "SAROSUSDT", "SATSUSDT", "SCAUSDT", "SCRTUSDT", "SCRUSDT", "SCUSDT", "SDUSDT",
            "SEILORUSDT", "SEIUSDT", "SENDUSDT", "SEORUSDT", "SERAPHUSDT", "SFUNDUSDT", "SHIBUSDT",
            "SHRAPUSDT", "SIDUSUSDT", "SISUSDT", "SLPUSDT", "SMILEUSDT", "SNXUSDT", "SOCIALUSDT",
            "SOLOUSDT", "SOLUSDT", "SOLVUSDT", "SONICUSDT", "SONUSDT", "SOSOUSDT", "SPARTAUSDT",
            "SPECUSDT", "SPELLUSDT", "SPXUSDT", "SQDUSDT", "SQRUSDT", "SQTUSDT", "SSVUSDT",
            "STARUSDT", "STATUSDT", "STETHUSDT", "STGUSDT", "STOPUSDT", "STREAMUSDT", "STRKUSDT",
            "STXUSDT", "SUIUSDT", "SUNDOGUSDT", "SUNUSDT", "SUPRAUSDT", "SUSHIUSDT", "SVLUSDT",
            "SWEATUSDT", "SWELLUSDT", "TAIKOUSDT", "TAIUSDT", "TAPUSDT", "TELUSDT", "TENETUSDT",
            "THETAUSDT", "THNUSDT", "THRUSTUSDT", "TIAUSDT", "TIMEUSDT", "TNSRUSDT", "TOKENUSDT",
            "TOMIUSDT", "TONUSDT", "TOSHIUSDT", "TRCUSDT", "TRUMPUSDT", "TRVLUSDT", "TRXUSDT",
            "TSTUSDT", "TURBOSUSDT", "TUSDUSDT", "TWTUSDT", "ULTIUSDT", "UMAUSDT", "UNIUSDT",
            "USDCUSDT", "USDDUSDT", "USDEUSDT", "USDTBUSDT", "USDYUSDT", "USTCUSDT", "UXLINKUSDT",
            "VANAUSDT", "VANRYUSDT", "VELARUSDT", "VELOUSDT", "VENOMUSDT", "VEXTUSDT", "VGXUSDT",
            "VINUUSDT", "VIRTUALUSDT", "VOXELUSDT", "VRUSDT", "VSCUSDT", "W3WUSDT", "WANUSDT",
            "WASUSDT", "WAVESUSDT", "WBTCUSDT", "WENUSDT", "WESTUSDT", "WIFUSDT", "WLDUSDT",
            "WLKNUSDT", "WOMUSDT", "WOOUSDT", "WSMUSDT", "XAIUSDT", "XAUTUSDT", "XCADUSDT",
            "XDCUSDT", "XECUSDT", "XEMUSDT", "XLMUSDT", "XMETAUSDT", "XMRUSDT", "XNGUSDT",
            "XRPUSDT", "XTZUSDT", "XVGNUSDT", "XVSUSDT", "YGGUSDT", "ZBCNUSDT", "ZBCUSDT",
            "ZCXUSDT", "ZENUSDT", "ZKJUSDT", "ZKSUSDT", "ZROUSDT", "ZRXUSDT"
        ]

    @staticmethod
    def _to_ccxt(symbol: str) -> str:
        if isinstance(symbol, str) and symbol.endswith("USDT") and "/" not in symbol:
            return symbol[:-4] + "/USDT"
        return symbol

    @staticmethod
    def _to_bybit(symbol: str) -> str:
        if isinstance(symbol, str) and "/" in symbol:
            return symbol.replace("/", "")
        return symbol

    def _format_pairs(self, pairs_bybit: List[str], fmt: Literal["ccxt", "bybit"]) -> List[str]:
        return pairs_bybit if fmt == "bybit" else [self._to_ccxt(p) for p in pairs_bybit]


def get_spot_usdt_pairs(fmt: Literal["ccxt", "bybit"] = "ccxt") -> List[str]:
    return PairManager().get_active_pairs(fmt=fmt)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pm = PairManager()
    pairs = pm.get_active_pairs(fmt="ccxt")
    print(f"Active USDT pairs ({len(pairs)}): {pairs[:15]}{'...' if len(pairs) > 15 else ''}")
