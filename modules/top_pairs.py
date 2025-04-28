# modules/top_pairs.py

import requests
import logging
import json
import os
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)

class PairManager:
    """
    Fetches and caches active USDT trading pairs from Bybit's spot symbols API.
    - Caches to a JSON file with 1-hour TTL.
    - Falls back to file or hardcoded list on failure.
    - Proactively expires the cache every hour in a background thread.
    """
    def __init__(self):
        self.cache_file = "pair_cache.json"
        self.cache_duration = timedelta(hours=1)
        self.fallback_file = "fallback_pairs.json"
        self.api_endpoint = "https://api.bybit.com/spot/v1/symbols"
        self.headers = {
            "User-Agent": "AI-Trading-Bot/1.0 (+https://github.com/your-repo)",
            "Accept": "application/json"
        }

        # Start a background timer to expire cache every hour
        self._start_cache_expiry_timer()

    def _start_cache_expiry_timer(self):
        """Daemon thread that removes the cache file every cache_duration."""
        def expire():
            try:
                if os.path.exists(self.cache_file):
                    os.remove(self.cache_file)
                    logger.info("Pair cache expired and removed proactively")
            except Exception as e:
                logger.error(f"Error expiring pair cache: {e}")
            finally:
                # Schedule next expiry
                threading.Timer(self.cache_duration.total_seconds(), expire).start()
        # Initial timer
        threading.Timer(self.cache_duration.total_seconds(), expire).start()

    def get_active_pairs(self) -> List[str]:
        """Main entry point: returns a list of active USDT pairs."""
        # Try fresh API data first
        try:
            pairs = self._fetch_api_pairs()
            if pairs:
                self._update_cache(pairs)
                logger.info(f"Loaded {len(pairs)} pairs from API")
                return pairs
        except Exception as e:
            logger.warning(f"API fetch failed: {e}")

        # Then try cached data
        cached = self._get_cached_pairs()
        if cached:
            logger.info(f"Loaded {len(cached)} pairs from cache")
            return cached

        # Then file fallback
        file_fb = self._get_file_fallback()
        if file_fb:
            logger.info(f"Loaded {len(file_fb)} pairs from fallback file")
            return file_fb

        # Finally, hardcoded fallback
        hard = self._get_hardcoded_fallback()
        logger.warning(f"Using hardcoded fallback pairs ({len(hard)} total)")
        return hard

    def _fetch_api_pairs(self) -> Optional[List[str]]:
        """Fetch pairs from Bybit API and validate the response."""
        resp = requests.get(self.api_endpoint, headers=self.headers, timeout=(3.05, 10))
        resp.raise_for_status()
        data = resp.json()
        if not self._validate_api_response(data):
            raise ValueError("Invalid API response structure")
        pairs = self._process_pairs(data["result"])
        return pairs

    def _validate_api_response(self, data: Dict) -> bool:
        """Ensure the API returned the expected keys and success code."""
        return data.get("ret_code") == 0 and "result" in data

    def _process_pairs(self, symbols: List[Dict]) -> List[str]:
        """Filter to USDT-quoted, Trading-status pairs and return their names."""
        valid = [
            sym["name"]
            for sym in symbols
            if sym.get("quoteCurrency") == "USDT" and sym.get("status") == "Trading"
        ]
        logger.debug(f"API returned {len(symbols)} symbols, {len(valid)} valid USDT pairs")
        return valid

    def _update_cache(self, pairs: List[str]) -> None:
        """Write the timestamped list of pairs to the cache file."""
        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "pairs": pairs
        }
        try:
            with open(self.cache_file, "w") as f:
                json.dump(payload, f)
        except Exception as e:
            logger.error(f"Failed to update pair cache: {e}")

    def _get_cached_pairs(self) -> Optional[List[str]]:
        """Return cached pairs if cache file exists and is still fresh."""
        try:
            if not os.path.exists(self.cache_file):
                return None
            with open(self.cache_file, "r") as f:
                data = json.load(f)
            ts = datetime.fromisoformat(data["timestamp"])
            if datetime.utcnow() - ts < self.cache_duration:
                return data["pairs"]
        except Exception as e:
            logger.error(f"Error reading pair cache: {e}")
        return None

    def _get_file_fallback(self) -> Optional[List[str]]:
        """Load fallback pairs from an external JSON file, if present."""
        try:
            if os.path.exists(self.fallback_file):
                with open(self.fallback_file, "r") as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error reading fallback file: {e}")
        return None

    def _get_hardcoded_fallback(self) -> List[str]:
        """Return a built-in list of popular USDT pairs."""
        # Only a small sample here — extend as needed.
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
        "VANAUSDT", "VANRYUSDT", "VELARUSDT", "VELOUSDT", "VENOMUSDT", "VEXTUSDT", "VICUSDT",
        "VINUUSDT", "VIRTUALUSDT", "VPADUSDT", "VPRUSDT", "VRAUSDT", "VRTXUSDT", "VVVUSDT",
        "WAVESUSDT", "WAXPUSDT", "WBTCUSDT", "WELLUSDT", "WEMIXUSDT", "WENUSDT", "WIFUSDT",
        "WLDUSDT", "WLKNUSDT", "WOOUSDT", "WUSDT", "WWYUSDT", "XAIUSDT", "XARUSDT", "XAVAUSDT",
        "XCADUSDT", "XDCUSDT", "XECUSDT", "XEMUSDT", "XETAUSDT", "XIONUSDT", "XLMUSDT", "XRPUSDT",
        "XTERUSDT", "XTZUSDT", "XUSDT", "XYMUSDT", "XZKUSDT", "YFIUSDT", "ZENDUSDT", "ZENTUSDT",
        "ZENUSDT", "ZEREBROUSDT", "ZEROUSDT", "ZETAUSDT", "ZEXUSDT", "ZIGUSDT", "ZILUSDT",
        "ZKJUSDT", "ZKLUSDT", "ZKUSDT", "ZRCUSDT", "ZROUSDT", "ZRXUSDT", "ZTXUSDT"
        ]


def get_spot_usdt_pairs() -> List[str]:
    """Convenience function to fetch active USDT pairs."""
    return PairManager().get_active_pairs()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pairs = get_spot_usdt_pairs()
    print(f"Active USDT pairs ({len(pairs)}): {pairs[:10]}{'...' if len(pairs)>10 else ''}")