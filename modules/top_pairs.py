# modules/top_pairs.py
import requests
import logging
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class PairManager:
    def __init__(self):
        self.cache_file = "pair_cache.json"
        self.cache_duration = timedelta(hours=1)
        self.fallback_file = "fallback_pairs.json"
        self.api_endpoint = "https://api.bybit.com/spot/v1/symbols"
        self.headers = {
            "User-Agent": "AI-Trading-Bot/1.0 (+https://github.com/your-repo)",
            "Accept": "application/json"
        }

    def get_active_pairs(self) -> List[str]:
        """Main entry point to get trading pairs with fallback strategy"""
        try:
            # Try to get fresh data
            pairs = self._fetch_api_pairs()
            if pairs:
                self._update_cache(pairs)
                return pairs
        except Exception as e:
            logger.warning(f"API fetch failed: {str(e)}")

        # Fallback strategies in order of priority
        return self._get_cached_pairs() or self._get_file_fallback() or self._get_hardcoded_fallback()

    def _fetch_api_pairs(self) -> Optional[List[str]]:
        """Fetch pairs from Bybit API with enhanced validation"""
        try:
            response = requests.get(
                self.api_endpoint,
                headers=self.headers,
                timeout=(3.05, 10)
            )
            response.raise_for_status()

            data = response.json()
            if not self._validate_api_response(data):
                raise ValueError("Invalid API response structure")

            return self._process_pairs(data.get("result", []))
        except Exception as e:
            logger.error(f"API request failed: {str(e)}")
            raise

    def _validate_api_response(self, data: Dict) -> bool:
        """Validate API response structure"""
        required_keys = ["ret_code", "result"]
        return all(key in data for key in required_keys) and data["ret_code"] == 0

    def _process_pairs(self, symbols: List[Dict]) -> List[str]:
        """Process and filter symbol data"""
        valid_pairs = [
            symbol["name"] for symbol in symbols
            if symbol.get("quoteCurrency") == "USDT" 
            and symbol.get("status") == "Trading"
        ]
        logger.info(f"Processed {len(valid_pairs)} active USDT pairs")
        return valid_pairs

    def _update_cache(self, pairs: List[str]) -> None:
        """Update local cache with timestamp"""
        cache_data = {
            "timestamp": datetime.now().isoformat(),
            "pairs": pairs
        }
        try:
            with open(self.cache_file, "w") as f:
                json.dump(cache_data, f)
        except Exception as e:
            logger.error(f"Cache update failed: {str(e)}")

    def _get_cached_pairs(self) -> Optional[List[str]]:
        """Get cached pairs if still valid"""
        try:
            if not os.path.exists(self.cache_file):
                return None

            with open(self.cache_file, "r") as f:
                data = json.load(f)

            cache_time = datetime.fromisoformat(data["timestamp"])
            if datetime.now() - cache_time < self.cache_duration:
                logger.info("Using valid cached pairs")
                return data["pairs"]
        except Exception as e:
            logger.error(f"Cache read failed: {str(e)}")
        return None

    def _get_file_fallback(self) -> Optional[List[str]]:
        """Get fallback pairs from external file"""
        try:
            if os.path.exists(self.fallback_file):
                with open(self.fallback_file, "r") as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"File fallback failed: {str(e)}")
        return None

    def _get_hardcoded_fallback(self) -> List[str]:
        """Final fallback to embedded pairs"""
        logger.warning("Using hardcoded fallback pairs")
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
    """Public interface for getting USDT pairs"""
    return PairManager().get_active_pairs()

if __name__ == "__main__":
    # Test script with verbose output
    import logging
    logging.basicConfig(level=logging.INFO)
    
    pairs = get_spot_usdt_pairs()
    print(f"Active USDT pairs ({len(pairs)}):")
    for pair in pairs[:10]:  # Print first 10 for demo
        print(pair)
    if len(pairs) > 10:
        print(f"... and {len(pairs)-10} more")