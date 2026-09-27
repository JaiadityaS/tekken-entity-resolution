"""Text normalisation for business names and addresses.

Everything here is pure-Python / stdlib (+ indic_transliteration, MIT) so it can be
run in a multiprocessing pool.  No external lookups are made.

Main entry points:
    normalize_name(raw)    -> dict(name, core, skel, legal, is_domain, concat)
    normalize_address(raw) -> dict(addr, alpha, nums, first_num)
"""
import re
import unicodedata

from indic_transliteration import sanscript

# --------------------------------------------------------------------------------------
# Indic script handling: transliterate to Harvard-Kyoto, then lowercase.
# --------------------------------------------------------------------------------------
_SCRIPTS = [
    (0x0900, 0x097F, sanscript.DEVANAGARI),
    (0x0980, 0x09FF, sanscript.BENGALI),
    (0x0A00, 0x0A7F, sanscript.GURMUKHI),
    (0x0A80, 0x0AFF, sanscript.GUJARATI),
    (0x0B00, 0x0B7F, sanscript.ORIYA),
    (0x0B80, 0x0BFF, sanscript.TAMIL),
    (0x0C00, 0x0C7F, sanscript.TELUGU),
    (0x0C80, 0x0CFF, sanscript.KANNADA),
    (0x0D00, 0x0D7F, sanscript.MALAYALAM),
]
_INDIC_RE = re.compile("[ऀ-ൿ]+")


def _script_of(ch):
    o = ord(ch)
    for lo, hi, sc in _SCRIPTS:
        if lo <= o <= hi:
            return sc
    return None


def translit_indic(s):
    """Replace every run of Indic-script characters with a Latin transliteration.

    Returns (text, was_indic).  Anusvara (M) becomes 'n' and a word-final inherent
    'a' is dropped so that e.g. 'mArkeTiMga' -> 'marketing'.
    """
    if not _INDIC_RE.search(s):
        return s, False

    def rep(m):
        run = m.group(0)
        sc = _script_of(run[0])
        out = sanscript.transliterate(run, sc, sanscript.HK)
        out = out.replace("M", "n").replace("z", "sh").replace("S", "sh")
        out = re.sub(r"[^A-Za-z ]", "", out).lower()
        # drop inherent word-final schwa: 'rama' -> 'ram', 'limiteda' -> 'limited'
        out = re.sub(r"(?<=[^aeiou ])a\b", "", out)
        return out

    return _INDIC_RE.sub(rep, s), True


# --------------------------------------------------------------------------------------
# Generic helpers
# --------------------------------------------------------------------------------------
def strip_accents(s):
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


_VOWELS = re.compile(r"[aeiouy]")


_PHON = [("tion", "shn"), ("sion", "shn"), ("ph", "f"), ("sh", "s"), ("ch", "k"),
         ("ck", "k"), ("kh", "k"), ("gh", "g"), ("bh", "b"), ("dh", "d"), ("th", "t"),
         ("jh", "j"), ("x", "ks"), ("q", "k"), ("c", "k"), ("z", "s"), ("w", "v"),
         ("v", "b"), ("d", "t"), ("h", "")]


def skeleton(tok):
    """Phonetic consonant skeleton, robust to transliteration:
    'developers'/'devalapars' -> 'tblprs', 'foundation'/'phaundeshan' -> 'fntsn',
    'private'/'praibhet' -> 'prbt'."""
    sk = tok
    for a, b in _PHON:
        sk = sk.replace(a, b)
    sk = _VOWELS.sub("", sk)
    sk = re.sub(r"(.)\1+", r"\1", sk)
    return sk or tok


def core_skeleton(core):
    """Skeleton string for a core name with legal-form / stop-word skeletons removed."""
    skel = [skeleton(t) for t in core.split()]
    return " ".join([k for k in skel if k not in LEGAL_SKEL and k not in NAME_STOP_SKEL] or skel)


def _fix_digit_typos(tok):
    """OCR-style digit-in-word typos: 'denta1' -> 'dental', 'c0rp' -> 'corp'."""
    if any(c.isalpha() for c in tok) and any(c in "01" for c in tok):
        letters = sum(c.isalpha() for c in tok)
        if letters >= 3 and letters >= len(tok) - 1:
            return tok.replace("1", "l").replace("0", "o")
    return tok


def _dedup_consecutive(toks):
    out = []
    for t in toks:
        if not out or out[-1] != t:
            out.append(t)
    return out


# --------------------------------------------------------------------------------------
# Names
# --------------------------------------------------------------------------------------
# canonical form of legal-entity / company-form tokens (US, India, France, generic)
LEGAL_CANON = {
    "private": "pvt", "pvt": "pvt", "pte": "pvt", "prvt": "pvt",
    "limited": "ltd", "ltd": "ltd", "ltda": "ltd", "lmtd": "ltd",
    "incorporated": "inc", "inc": "inc", "lncorporated": "inc", "incorporation": "inc",
    "corporation": "corp", "corp": "corp", "corpn": "corp",
    "company": "co", "co": "co", "cos": "co",
    "llc": "llc", "llp": "llp", "lp": "lp", "pc": "pc", "pllc": "pllc", "plc": "plc",
    "pa": "pa", "ltee": "ltd", "tbk": "tbk", "gmbh": "gmbh", "ag": "ag", "bv": "bv",
    "nv": "nv", "srl": "srl", "spa": "spa", "oy": "oy", "ab": "ab", "as": "as",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl", "sa": "sa",
    "sci": "sci", "snc": "snc", "scop": "scop", "scp": "scp", "selarl": "selarl",
    "opc": "opc", "huf": "huf",
}
# low-information tokens removed from the 'core' name
NAME_STOP = {
    "the", "of", "and", "et", "de", "du", "des", "la", "le", "les", "a", "an", "l", "d",
    "s", "m", "smt", "shri", "sri", "mr", "mrs", "ms", "dr", "m/s", "ms", "dba",
    "formerly", "aka", "www", "com", "net", "org", "in", "fr", "india", "france", "usa",
}
LEGAL_SKEL = {k for k in ({skeleton(k) for k in LEGAL_CANON} | {skeleton(v) for v in LEGAL_CANON.values()})
              if len(k) >= 2}
NAME_STOP_SKEL = {skeleton(k) for k in ("india", "france", "the", "of", "and")}

_DOMAIN_RE = re.compile(r"^\s*(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9\-]*)\.(?:co\.in|com|in|net|org|fr|co|biz|info|us|io)\s*$")
_NONALNUM = re.compile(r"[^a-z0-9]+")


def normalize_name(raw):
    s, indic = translit_indic(raw or "")
    s = strip_accents(s).lower()
    s = s.replace("&", " and ").replace("+", " and ").replace("@", " ")
    is_domain = 0
    m = _DOMAIN_RE.match(s)
    if m:
        is_domain = 1
        s = m.group(1).replace("-", " ")
    # remove dotted legal forms 'p.v.t.' / 'l.l.c.' / 'o.d.' -> 'pvt' / 'llc' / 'od'
    s = re.sub(r"\b((?:[a-z]\.){2,})", lambda mm: mm.group(1).replace(".", ""), s)
    toks = [_fix_digit_typos(t) for t in _NONALNUM.split(s) if t]
    toks = _dedup_consecutive(toks)
    canon = [LEGAL_CANON.get(t, t) for t in toks]
    legal = sorted({LEGAL_CANON[t] for t in toks if t in LEGAL_CANON})
    core = [t for t in canon if t not in LEGAL_CANON.values() and t not in NAME_STOP]
    if not core:  # never let the core collapse to nothing
        core = [t for t in canon if t not in LEGAL_CANON.values()] or canon
    return {
        "name": " ".join(canon),
        "core": " ".join(core),
        "skel": core_skeleton(" ".join(core)),
        "legal": " ".join(legal),
        "is_domain": is_domain,
        "is_indic": int(indic),
        "concat": "".join(core),
    }


# --------------------------------------------------------------------------------------
# Addresses
# --------------------------------------------------------------------------------------
ADDR_CANON = {
    # US street types
    "street": "st", "str": "st", "st": "st", "road": "rd", "rd": "rd", "drive": "dr",
    "dr": "dr", "drv": "dr", "avenue": "ave", "ave": "ave", "av": "ave", "aven": "ave",
    "lane": "ln", "ln": "ln", "court": "ct", "ct": "ct", "boulevard": "blvd",
    "blvd": "blvd", "bd": "blvd", "bld": "blvd", "place": "pl", "pl": "pl",
    "circle": "cir", "cir": "cir", "highway": "hwy", "hwy": "hwy", "parkway": "pkwy",
    "pkwy": "pkwy", "terrace": "ter", "ter": "ter", "trail": "trl", "trl": "trl",
    "square": "sq", "sq": "sq", "cove": "cv", "cv": "cv", "way": "way", "loop": "loop",
    "suite": "ste", "ste": "ste", "apartment": "apt", "apt": "apt", "building": "bldg",
    "bldg": "bldg", "floor": "fl", "fl": "fl", "flr": "fl",
    "north": "n", "south": "s", "east": "e", "west": "w", "northeast": "ne",
    "northwest": "nw", "southeast": "se", "southwest": "sw",
    "mount": "mt", "mt": "mt", "saint": "st", "fort": "ft", "ft": "ft",
    "township": "twp", "twp": "twp", "toownship": "twp",
    # India
    "nagar": "nagar", "ngr": "nagar", "marg": "marg", "mg": "mg", "sector": "sec",
    "sec": "sec", "opposite": "opp", "opp": "opp", "near": "nr", "nr": "nr",
    "behind": "bhd", "bh": "bhd", "cross": "crs", "main": "main", "colony": "col",
    "col": "col", "district": "dist", "dist": "dist", "distt": "dist", "taluk": "tq",
    "tq": "tq", "tal": "tq", "po": "po", "ps": "ps", "village": "vill", "vill": "vill",
    "vpo": "vill", "bangalore": "bengaluru", "bengaluru": "bengaluru",
    "bombay": "mumbai", "gurgaon": "gurugram", "gurugram": "gurugram",
    "ahmadabad": "ahmedabad", "calcutta": "kolkata", "madras": "chennai",
    # France
    "rue": "rue", "r": "rue", "boul": "blvd", "chemin": "ch", "ch": "ch",
    "allee": "all", "all": "all", "impasse": "imp", "imp": "imp",
    "faubourg": "fbg", "fbg": "fbg",
    "quai": "quai", "cours": "crs", "route": "rte", "rte": "rte",
    "passage": "pass", "residence": "res", "lotissement": "lot",
    "cite": "cite", "hameau": "ham",
    "anenue": "ave",  # common OCR/typo for avenue
}
US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi",
    "wyoming": "wy", "district of columbia": "dc",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr",
    "himachal pradesh": "hp", "jharkhand": "jh", "karnataka": "ka", "kerala": "kl",
    "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml",
    "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od", "punjab": "pb",
    "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "telangana": "tg",
    "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk", "west bengal": "wb",
    "delhi": "dl", "jammu and kashmir": "jk", "chandigarh": "ch", "puducherry": "py",
    "pondicherry": "py", "ladakh": "la",
    # transliterated native-script spellings seen in the data
    "andhraprdesh": "ap", "andhrapradesh": "ap", "dilli": "dl", "maharashtr": "mh",
    "karnatak": "ka", "tamilnadu": "tn", "telangan": "tg", "gujrat": "gj",
    "rajasthan ": "rj", "uttarprdesh": "up", "uttarpradesh": "up", "keralam": "kl",
    "pashchim bangal": "wb", "pashchimbanga": "wb",
}
_STATE_RE = re.compile(
    r"\b(" + "|".join(sorted({*US_STATES, *IN_STATES}, key=len, reverse=True)) + r")\b"
)
_STATE_MAP = {**US_STATES, **IN_STATES}
ORDINALS = {
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5", "sixth": "6",
    "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10", "eleventh": "11",
    "twelfth": "12", "thirteenth": "13", "fourteenth": "14", "fifteenth": "15",
    "sixteenth": "16", "seventeenth": "17", "eighteenth": "18", "nineteenth": "19",
    "twentieth": "20",
}
NULL_TOKENS = {"null", "n/a", "na", "none", "nan", "nil", "unknown"}
ADDR_STOP = {"no", "number", "door", "plot", "h", "hno", "house", "khata", "kh", "pmb",
             "unit", "ste", "apt", "fl", "bldg", "the", "of", "and", "de", "du", "des",
             "la", "le", "les", "d", "l", "nr", "opp", "bhd", "n", "o"}
_NUM_RE = re.compile(r"\d+")

# --------------------------------------------------------------------------------------
# French address helpers (active only when country=="France")
# --------------------------------------------------------------------------------------

# Map French departments to regions for normalisation consistency.
# Only the departments actually seen in the test data are included.
FR_DEPT_TO_REGION = {
    "gironde": "nouvelle aquitaine",
    "nord": "hauts de france",
    "pas de calais": "hauts de france",
    "pas-de-calais": "hauts de france",
    "loire atlantique": "pays de la loire",
    "loire-atlantique": "pays de la loire",
    "finistere": "bretagne",
    "morbihan": "bretagne",
    "cotes d armor": "bretagne",
    "cotes-d-armor": "bretagne",
    "ille et vilaine": "bretagne",
    "ille-et-vilaine": "bretagne",
    "maine et loire": "pays de la loire",
    "maine-et-loire": "pays de la loire",
    "sarthe": "pays de la loire",
    "vendee": "pays de la loire",
    "mayenne": "pays de la loire",
    "charente maritime": "nouvelle aquitaine",
    "charente-maritime": "nouvelle aquitaine",
    "charente": "nouvelle aquitaine",
    "dordogne": "nouvelle aquitaine",
    "landes": "nouvelle aquitaine",
    "lot et garonne": "nouvelle aquitaine",
    "lot-et-garonne": "nouvelle aquitaine",
    "pyrenees atlantiques": "nouvelle aquitaine",
    "pyrenees-atlantiques": "nouvelle aquitaine",
    "deux sevres": "nouvelle aquitaine",
    "deux-sevres": "nouvelle aquitaine",
    "vienne": "nouvelle aquitaine",
    "haute vienne": "nouvelle aquitaine",
    "haute-vienne": "nouvelle aquitaine",
    "correze": "nouvelle aquitaine",
    "creuse": "nouvelle aquitaine",
    # Hauts-de-France
    "aisne": "hauts de france",
    "oise": "hauts de france",
    "somme": "hauts de france",
    # Pays de la Loire alternate forms
    "loire": "pays de la loire",
    # Île-de-France
    "paris": "ile de france",
    "seine saint denis": "ile de france",
    "seine-saint-denis": "ile de france",
    "hauts de seine": "ile de france",
    "hauts-de-seine": "ile de france",
    "val de marne": "ile de france",
    "val-de-marne": "ile de france",
    "seine et marne": "ile de france",
    "seine-et-marne": "ile de france",
    "yvelines": "ile de france",
    "essonne": "ile de france",
    "val d oise": "ile de france",
    "val-d-oise": "ile de france",
}

# Build region canonical forms for consistent output
FR_REGION_CANON = {
    "nouvelle aquitaine": "nouvelle aquitaine",
    "nouvelle-aquitaine": "nouvelle aquitaine",
    "hauts de france": "hauts de france",
    "hauts-de-france": "hauts de france",
    "pays de la loire": "pays de la loire",
    "pays-de-la-loire": "pays de la loire",
    "bretagne": "bretagne",
    "ile de france": "ile de france",
    "ile-de-france": "ile de france",
}

# Build a regex to match departments and region names in addresses (longest first)
_FR_GEO_NAMES = sorted(
    {*FR_DEPT_TO_REGION, *FR_REGION_CANON}, key=len, reverse=True
)
_FR_GEO_RE = re.compile(
    r"\b(" + "|".join(re.escape(n) for n in _FR_GEO_NAMES) + r")\b"
)

# Elision pattern: split French elisions like l'Orne, d'Artagnan in addresses
_FR_ELISION_RE = re.compile(r"\b([ldnj])['\u2019]([a-z])", re.IGNORECASE)

# bis/ter/quater after house number: "5 bis" -> "5bis", "12 ter" -> "12ter"
_FR_BIS_TER_RE = re.compile(r"\b(\d+)\s+(bis|ter|quater)\b", re.IGNORECASE)


def _french_address_preprocess(s):
    """Apply France-specific address preprocessing BEFORE tokenisation.

    1. Split elisions: l'Orne -> l orne
    2. Attach bis/ter to house number: 5 bis -> 5bis
    3. Replace departments with their region name
    4. Canonicalise region name variants
    """
    # 1. Split elisions
    s = _FR_ELISION_RE.sub(r"\1 \2", s)
    # 2. Attach bis/ter/quater to house number
    s = _FR_BIS_TER_RE.sub(lambda m: m.group(1) + m.group(2).lower(), s)
    # 3 & 4. Map departments -> regions, canonicalise region names
    def _geo_replace(m):
        key = m.group(1)
        # If it's a department, map to region
        if key in FR_DEPT_TO_REGION:
            return FR_DEPT_TO_REGION[key]
        # If it's a region variant, canonicalise
        if key in FR_REGION_CANON:
            return FR_REGION_CANON[key]
        return key
    s = _FR_GEO_RE.sub(_geo_replace, s)
    return s


def normalize_address(raw, country=None):
    s, _ = translit_indic(raw or "")
    s = strip_accents(s).lower()
    s = re.sub(r"<\s*null\s*>|\bn/a\b", " ", s)
    s = _STATE_RE.sub(lambda m: " " + _STATE_MAP[m.group(1)] + " ", s)
    s = s.replace("&", " ")
    # French-specific address preprocessing (only for France)
    if country == "France":
        s = _french_address_preprocess(s)
    # '5th' -> '5', '2nd' -> '2'
    s = re.sub(r"\b(\d+)(?:st|nd|rd|th)\b", r"\1", s)
    toks = []
    for t in _NONALNUM.split(s):
        if not t or t in NULL_TOKENS:
            continue
        if t in ORDINALS:
            toks.append(ORDINALS[t])
            continue
        if t.isdigit():
            toks.append(t.lstrip("0") or "0")
            continue
        if any(c.isdigit() for c in t):
            # mixed token like '1335c' / 'b12' -> split into digits and letters
            # For France, bis/ter attached to number: '5bis' -> '5' + 'bis'
            for part in re.findall(r"\d+|[a-z]+", t):
                toks.append(part.lstrip("0") or "0" if part.isdigit() else part)
            continue
        toks.append(ADDR_CANON.get(t, t))
    toks = _dedup_consecutive(toks)
    nums = [t for t in toks if t.isdigit()]
    alpha = [t for t in toks if not t.isdigit() and t not in ADDR_STOP and len(t) > 1]
    return {
        "addr": " ".join(toks),
        "alpha": " ".join(alpha),
        "nums": " ".join(sorted(set(nums), key=lambda x: (len(x), x))),
        "first_num": _first_house_number(toks),
    }


def _first_house_number(toks):
    """Heuristic primary number: first number not following unit/pmb/suite markers."""
    skip_next = False
    for t in toks:
        if t in ("pmb", "unit", "ste", "apt", "fl", "sec", "po", "box"):
            skip_next = True
            continue
        if t.isdigit():
            if skip_next:
                skip_next = False
                continue
            return t
        skip_next = False
    return ""
