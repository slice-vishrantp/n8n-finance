"""
credit_ratings_scraper.py
------------------------
Fetches credit ratings for a list of companies from public rating agencies by:
1) CRISIL: calling the public "Credit Rating List" JSON endpoint behind
   https://www.crisilratings.com/en/home/our-business/ratings/credit-ratings-list.html
   and optionally matching a provided company list locally.
2) Other agencies: (optional) search + scrape via DuckDuckGo HTML + public pages.

Notes / caveats:
- Some environments do TLS inspection; use --ca-bundle /path/to/corp-ca.pem, or last resort --insecure.
- The CRISIL "credit rating list" endpoint paginates with:
    start=<offset>, limit=<offset+page_size>   (limit behaves like an END index, not page size)
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import re
import time
import urllib.parse
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


try:
    import requests  # type: ignore
    from requests.adapters import HTTPAdapter  # type: ignore
    from requests.exceptions import SSLError  # type: ignore
    from urllib3.exceptions import InsecureRequestWarning  # type: ignore
    from urllib3.util.retry import Retry  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # type: ignore

try:
    import pdfplumber  # type: ignore
except Exception:  # pragma: no cover
    pdfplumber = None  # type: ignore

try:
    from PyPDF2 import PdfReader  # type: ignore
except Exception:  # pragma: no cover
    PdfReader = None  # type: ignore

try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text  # type: ignore
except Exception:  # pragma: no cover
    pdfminer_extract_text = None  # type: ignore


LONG_TERM_RATING_RE = re.compile(
    r"\b(AAA|AA\+|AA-|AA|A\+|A-|A|BBB\+|BBB-|BBB|BB\+|BB-|BB|B\+|B-|B|C|D)\b",
    re.IGNORECASE,
)

# Common short-term scales used across agencies (not exhaustive).
SHORT_TERM_RATING_RE = re.compile(
    # NOTE: Do NOT use \b boundaries here: tokens like "A1+" end with a non-word char (+),
    # and `\b` would cause us to incorrectly match "A1" instead of "A1+".
    r"(?<![A-Z0-9])("
    r"A1\s*\+|A1|A2\s*\+|A2|A3\s*\+|A3|A4\s*\+|A4|"
    r"P1\s*\+|P1|P2|P3|P4|"
    r"PR1\s*\+|PR1|PR2|PR3|PR4|"
    r"F1\s*\+|F1|F2|F3"
    r")(?![A-Z0-9])",
    re.IGNORECASE,
)

OUTLOOK_RE = re.compile(r"\b(Stable|Positive|Negative|Developing|Watch)\b", re.IGNORECASE)


def _require_requests() -> None:
    if requests is None:
        raise RuntimeError(
            "Missing dependency: requests. Install with:\n"
            "  python3 -m pip install requests\n"
        )


def make_session(insecure: bool = False, ca_bundle: Optional[str] = None) -> "requests.Session":
    """
    Create a shared session with retries/backoff.

    If your environment does TLS inspection (common in corporate networks), prefer:
      --ca-bundle /path/to/corp-ca.pem
    Use --insecure only as a last resort (disables TLS verification).
    """
    _require_requests()
    assert requests is not None  # mypy

    s = requests.Session()
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        backoff_factor=0.7,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)

    if insecure:
        s.verify = False
        # Silence urllib3 warnings when the user explicitly requests --insecure.
        try:
            requests.packages.urllib3.disable_warnings(InsecureRequestWarning)  # type: ignore[attr-defined]
        except Exception:
            pass
    elif ca_bundle:
        s.verify = ca_bundle

    return s


def fetch_text(url: str, timeout_s: int = 20, session: Optional["requests.Session"] = None) -> str:
    _require_requests()
    assert requests is not None  # mypy
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/121.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    s = session or requests.Session()
    resp = s.get(url, headers=headers, timeout=timeout_s)
    resp.raise_for_status()
    return resp.text


def fetch_text_with_retries(
    url: str,
    *,
    session: Optional["requests.Session"] = None,
    timeout_s: int = 20,
    attempts: int = 3,
    backoff_s: float = 0.8,
) -> str:
    """
    Like fetch_text(), but retries on transient failures (HTTP 429/5xx, timeouts, connection errors).
    """
    _require_requests()
    assert requests is not None  # mypy

    last_err: Optional[BaseException] = None
    for i in range(max(1, int(attempts))):
        try:
            return fetch_text(url, timeout_s=timeout_s, session=session)
        except Exception as e:
            last_err = e
            status_code: Optional[int] = None
            try:
                if isinstance(e, requests.exceptions.HTTPError) and getattr(e, "response", None) is not None:
                    status_code = int(e.response.status_code)  # type: ignore[union-attr]
            except Exception:
                status_code = None

            retryable_http = status_code in {408, 425, 429, 500, 502, 503, 504}
            retryable_exc = isinstance(
                e,
                (
                    requests.exceptions.Timeout,
                    requests.exceptions.ConnectionError,
                    requests.exceptions.ChunkedEncodingError,
                ),
            )

            if i >= (max(1, int(attempts)) - 1):
                break
            if not (retryable_http or retryable_exc):
                break
            time.sleep(max(0.0, float(backoff_s)) * (2**i))

    assert last_err is not None
    raise last_err


def fetch_text_crisil_doc(
    url: str,
    *,
    session: Optional["requests.Session"] = None,
    timeout_s: int = 30,
    attempts: int = 2,
) -> str:
    """
    Fetch CRISIL RatingDocs (`/mnt/winshare/...`) with host fallback.

    In practice, the same RatingDocs path can be accessible on BOTH:
    - https://www.crisilratings.com
    - https://www.crisil.com

    We try the given URL first; on 404 we retry by swapping hosts.
    """
    _require_requests()
    assert requests is not None  # mypy

    def _swap_host(u: str) -> Optional[str]:
        try:
            parsed = urllib.parse.urlparse(u)
            if not parsed.path.startswith("/mnt/"):
                return None
            host = parsed.netloc.lower()
            if host.endswith("crisilratings.com"):
                alt_host = urllib.parse.urlparse(CRISIL_DOCS_BASE).netloc
            elif host.endswith("crisil.com"):
                alt_host = urllib.parse.urlparse(CRISIL_BASE).netloc
            else:
                return None
            alt = parsed._replace(netloc=alt_host, scheme="https")
            return urllib.parse.urlunparse(alt)
        except Exception:
            return None

    last_err: Optional[BaseException] = None
    for i in range(max(1, int(attempts))):
        try:
            return fetch_text(url=url, timeout_s=timeout_s, session=session)
        except Exception as e:
            last_err = e
            try:
                if isinstance(e, requests.exceptions.HTTPError) and getattr(e, "response", None) is not None:
                    if int(e.response.status_code) == 404:  # type: ignore[union-attr]
                        alt = _swap_host(url)
                        if alt and alt != url:
                            url = alt
                            continue
            except Exception:
                pass
            break

    assert last_err is not None
    raise last_err


def icra_rating_details_url(company_id: str, company_name: str) -> str:
    q = urllib.parse.urlencode(
        {"CompanyId": str(company_id).strip(), "CompanyName": str(company_name).strip()},
        quote_via=urllib.parse.quote,
    )
    return f"{ICRA_BASE}/Rating/RatingDetails?{q}"


def icra_search_companies(term: str, *, session: "requests.Session") -> List[Dict[str, str]]:
    """
    Resolve CompanyId from a company name using ICRA's autocomplete endpoint.
    Returns items like: {"id": "15410", "label": "Tata Capital Housing Finance Limited"}
    """
    url = f"{ICRA_BASE}/Rating/GetRatingCompanys"
    resp = session.post(
        url,
        data={"Term": term},
        headers={"User-Agent": "Mozilla/5.0", "X-Requested-With": "XMLHttpRequest"},
        timeout=30,
    )
    resp.raise_for_status()
    items = resp.json()
    out: List[Dict[str, str]] = []
    if isinstance(items, list):
        for it in items:
            if isinstance(it, dict) and it.get("id") and it.get("label"):
                out.append({"id": str(it["id"]), "label": str(it["label"])})
    return out


def icra_pick_best_company(term: str, items: Sequence[Dict[str, str]]) -> Optional[Dict[str, str]]:
    if not items:
        return None
    t = _norm_company_name_relaxed(term)

    def score(it: Dict[str, str]) -> int:
        label = it.get("label", "")
        l = _norm_company_name_relaxed(label)
        if l == t:
            return 100
        if t and t in l:
            return 80
        return 0

    return sorted(items, key=score, reverse=True)[0]


def fetch_icra_rating_details(
    *,
    session: "requests.Session",
    company_id: str,
    company_name: str,
) -> Dict[str, str]:
    """
    Fetch ICRA rating details for a companyId + companyName.
    Extracts:
    - long_term_rating: best long-term token found on page
    - short_term_rating: rating for 'Commercial Paper' if available, else best ST token
    - updated_on: best past date found on page
    """
    url = icra_rating_details_url(company_id, company_name)
    html = fetch_text(url, session=session, timeout_s=30)
    text = html_to_text(html)

    # Prefer parsing instrument list items from the DOM to avoid picking up unrelated '[ICRA]' tokens
    # elsewhere on the page (e.g. other entities shown in site widgets).
    instruments: List[Tuple[str, str]] = []
    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(html, "html.parser")
        for li in soup.find_all("li"):
            # On ICRA RatingDetails pages, instrument ratings are in plain <li> elements with no class.
            # This avoids pulling in navigation/widgets/rationales which use classed list items.
            if li.get("class") is not None:
                continue
            t = li.get_text(" ", strip=True)
            if "[ICRA]" not in t:
                continue
            m = re.search(r"^(?P<instr>.+?)\\s*\\[ICRA\\]\\s*(?P<rating>[A-Za-z0-9+\\-]+)", t, flags=re.I)
            if not m:
                continue
            instr = re.sub(r"\s+", " ", (m.group("instr") or "")).strip()
            rating = re.sub(r"\s+", "", (m.group("rating") or "")).upper()
            if instr and rating:
                instruments.append((instr, rating))
    except Exception:
        instruments = []

    # Fallback: instrument lines in flattened text (less precise but works if DOM changes)
    if not instruments:
        instr_pat = re.compile(
            r"(?P<instr>[A-Za-z0-9&/().,\\s-]+?)\\s*\\[ICRA\\]\\s*(?P<rating>[A-Za-z0-9+\\-]+)",
            flags=re.IGNORECASE,
        )
        for m in instr_pat.finditer(text):
            instr = re.sub(r"\s+", " ", (m.group("instr") or "")).strip()
            rating = re.sub(r"\s+", "", (m.group("rating") or "")).upper()
            if instr and rating:
                instruments.append((instr, rating))

    short_term = ""
    for instr, rating in instruments:
        if instr.lower() == "commercial paper":
            short_term = rating
            break
    if not short_term:
        short_term = parse_rating_tokens_simple(text).get("short_term_rating", "")

    # Prefer long-term rating from instrument list (avoids picking tokens from unrelated parts of the page)
    def _lt_rank(tok: str) -> int:
        order = [
            "AAA",
            "AA+",
            "AA",
            "AA-",
            "A+",
            "A",
            "A-",
            "BBB+",
            "BBB",
            "BBB-",
            "BB+",
            "BB",
            "BB-",
            "B+",
            "B",
            "B-",
            "C",
            "D",
        ]
        try:
            return 100 - order.index(tok)
        except ValueError:
            return 0

    lt_vocab = {
        "AAA",
        "AA+",
        "AA",
        "AA-",
        "A+",
        "A",
        "A-",
        "BBB+",
        "BBB",
        "BBB-",
        "BB+",
        "BB",
        "BB-",
        "B+",
        "B",
        "B-",
        "C",
        "D",
    }
    long_term = ""
    for instr, rating in instruments:
        if instr.lower() == "commercial paper":
            continue
        if rating not in lt_vocab:
            continue
        if not long_term or _lt_rank(rating) > _lt_rank(long_term):
            long_term = rating
    if not long_term:
        # Fallback: extract from known instrument labels in the flattened text.
        for label in (
            "Long Term-Fund-Based/Non-Fund Based",
            "Non-Convertible Debentures",
            "Subordinated Debt",
            "Retail Bonds",
        ):
            m = re.search(
                re.escape(label) + r"\\s*\\[ICRA\\]\\s*([A-Za-z0-9+\\-]+)",
                text,
                flags=re.IGNORECASE,
            )
            if m:
                cand = re.sub(r"\\s+", "", m.group(1)).upper()
                if cand in lt_vocab:
                    long_term = cand
                    break
    if not long_term:
        long_term = parse_rating_tokens_simple(text).get("long_term_rating", "")
    updated_on = parse_date(text) or ""

    return {
        "agency": "ICRA",
        "company_id": str(company_id),
        "company_name": str(company_name),
        "matched_company_name": str(company_name),
        "long_term_rating": long_term,
        "short_term_rating": short_term,
        "updated_on": updated_on,
        "source_url": url,
        "date_retrieved": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "status": "ok" if (long_term or short_term) else "parsed_no_rating",
    }


def html_to_text(html: str) -> str:
    """
    Convert HTML to plain-ish text.
    If BeautifulSoup is available, we use it; otherwise do a lightweight strip.
    """
    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        return re.sub(r"\s+", " ", text)
    except Exception:
        text = re.sub(r"<[^>]+>", " ", html)
        return re.sub(r"\s+", " ", text).strip()


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """
    Extract text from a PDF bytes blob.
    Uses a fallback chain: pdfplumber -> pdfminer -> PyPDF2.
    """
    if not pdf_bytes:
        return ""

    # 1) pdfplumber (often best at tables)
    if pdfplumber is not None:
        try:
            out: List[str] = []
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for p in pdf.pages:
                    out.append(p.extract_text() or "")
            txt = "\n".join(out).strip()
            if txt:
                return txt
        except Exception:
            pass

    # 2) pdfminer (robust)
    if pdfminer_extract_text is not None:
        try:
            txt = (pdfminer_extract_text(io.BytesIO(pdf_bytes)) or "").strip()
            if txt:
                return txt
        except Exception:
            pass

    # 3) PyPDF2 (fast fallback)
    if PdfReader is not None:
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            out = []
            for p in reader.pages:
                out.append(p.extract_text() or "")
            txt = "\n".join(out).strip()
            if txt:
                return txt
        except Exception:
            pass

    return ""


CARE_LONG_TERM_RE = re.compile(
    r"\bCARE\s+"
    r"(?P<rating>(?:AAA|AA|A|BBB|BB|B|C|D)(?:\+|-)?)(?:\s*;?\s*(?P<outlook>Stable|Positive|Negative|Developing))?\b",
    re.IGNORECASE,
)
CARE_SHORT_TERM_RE = re.compile(
    # Allow optional whitespace before '+' because PDF text extraction sometimes yields "A1 +"
    r"(?<![A-Z0-9])CARE\s+(?P<rating>(?:A1\s*\+|A1|A2\s*\+|A2|A3\s*\+|A3|A4|A5|PR1\s*\+|PR1|PR2|PR3|PR4|PR5))(?![A-Z0-9])",
    re.IGNORECASE,
)


def parse_care_ratings(text: str) -> Dict[str, str]:
    """
    Extract best-effort CARE long-term, short-term, outlook from PDF text.
    """
    txt = (text or "").replace("\u00a0", " ")
    lt: List[str] = []
    st: List[str] = []
    outlook = ""

    for m in CARE_LONG_TERM_RE.finditer(txt):
        r = (m.group("rating") or "").upper().strip()
        r = re.sub(r"\s+", "", r)
        if r:
            lt.append(r)
        o = (m.group("outlook") or "").strip()
        if o and not outlook:
            outlook = o.title()

    for m in CARE_SHORT_TERM_RE.finditer(txt):
        r = (m.group("rating") or "").upper().strip()
        r = re.sub(r"\s+", "", r)
        if r:
            st.append(r)

    # De-dupe preserving order
    def _uniq(xs: List[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for x in xs:
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

    lt_u = _uniq(lt)
    st_u = _uniq(st)

    # Prefer the "best"/highest rating when multiple are present.
    # CARE LT scale: AAA > AA+ > AA > AA- > A+ > ... > D
    lt_order = [
        "AAA",
        "AA+",
        "AA",
        "AA-",
        "A+",
        "A",
        "A-",
        "BBB+",
        "BBB",
        "BBB-",
        "BB+",
        "BB",
        "BB-",
        "B+",
        "B",
        "B-",
        "C",
        "D",
    ]
    lt_rank = {v: i for i, v in enumerate(lt_order)}
    best_lt = ""
    if lt_u:
        best_lt = sorted(lt_u, key=lambda r: lt_rank.get(r, 10_000))[0]

    # CARE ST scale: A1+ > A1 > A2+ > A2 > A3+ > A3 > A4 > A5; PR1+ > ... > PR5
    st_order = ["A1+", "A1", "A2+", "A2", "A3+", "A3", "A4", "A5", "PR1+", "PR1", "PR2", "PR3", "PR4", "PR5"]
    st_rank = {v: i for i, v in enumerate(st_order)}
    best_st = ""
    if st_u:
        best_st = sorted(st_u, key=lambda r: st_rank.get(r, 10_000))[0]

    return {
        "long_term_rating": best_lt,
        "short_term_rating": best_st,
        "outlook": outlook,
    }


def fetch_bytes(url: str, session: "requests.Session", timeout_s: int = 30) -> bytes:
    r = session.get(url, timeout=timeout_s, allow_redirects=True)
    r.raise_for_status()
    return r.content or b""


def care_ratings_from_url(
    *,
    session: "requests.Session",
    url: str,
    company_name: str = "",
) -> Dict[str, str]:
    """
    Parse a CARE Ratings press release PDF URL.
    Example URL:
      https://www.careratings.com/upload/CompanyFiles/PR/202510161043_Jindal_Stainless_Limited.pdf
    """
    retrieved_at = dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    try:
        pdf_bytes = fetch_bytes(url, session=session, timeout_s=45)
        text = extract_pdf_text(pdf_bytes)
        pr = parse_care_ratings(text)
        updated_on = parse_date(text) or ""
        return {
            "agency": "CARE Ratings",
            "company_name": company_name or "",
            "company_id": "",
            "long_term_rating": pr.get("long_term_rating") or "",
            "short_term_rating": pr.get("short_term_rating") or "",
            "outlook": pr.get("outlook") or "",
            "updated_on": updated_on,
            "source_url": url,
            "date_retrieved": retrieved_at,
            "status": "ok" if (pr.get("long_term_rating") or pr.get("short_term_rating")) else "parsed_no_signal",
        }
    except Exception as e:
        return {
            "agency": "CARE Ratings",
            "company_name": company_name or "",
            "company_id": "",
            "long_term_rating": "",
            "short_term_rating": "",
            "outlook": "",
            "updated_on": "",
            "source_url": url,
            "date_retrieved": retrieved_at,
            "status": f"error:{type(e).__name__}",
        }


def icra_rationale_from_id(
    *,
    session: "requests.Session",
    rationale_id: str,
    company_name: str = "",
) -> Dict[str, str]:
    """
    Fetch and parse an ICRA rationale report by Id.
    Example:
      https://www.icra.in/Rationale/ShowRationaleReport?Id=134727
    Underlying PDF:
      https://www.icra.in/Rating/ShowRationalReportFilePdf/134727
    """
    rid = str(rationale_id).strip()
    retrieved_at = dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    if not rid.isdigit():
        return {
            "agency": "ICRA",
            "company_name": company_name or "",
            "company_id": "",
            "long_term_rating": "",
            "short_term_rating": "",
            "outlook": "",
            "updated_on": "",
            "source_url": f"{ICRA_BASE}/Rationale/ShowRationaleReport?Id={rid}",
            "date_retrieved": retrieved_at,
            "status": "error:invalid_id",
        }

    pdf_url = f"{ICRA_BASE}/Rating/ShowRationalReportFilePdf/{rid}"
    try:
        pdf_bytes = fetch_bytes(pdf_url, session=session, timeout_s=45)
        text = extract_pdf_text(pdf_bytes)

        # Parse ratings
        # Use existing vocab/ranking from fetch_icra_rating_details, but derive from PDF text.
        lt_vocab = {
            "AAA",
            "AA+",
            "AA",
            "AA-",
            "A+",
            "A",
            "A-",
            "BBB+",
            "BBB",
            "BBB-",
            "BB+",
            "BB",
            "BB-",
            "B+",
            "B",
            "B-",
            "C",
            "D",
        }

        def _lt_rank(tok: str) -> int:
            order = [
                "AAA",
                "AA+",
                "AA",
                "AA-",
                "A+",
                "A",
                "A-",
                "BBB+",
                "BBB",
                "BBB-",
                "BB+",
                "BB",
                "BB-",
                "B+",
                "B",
                "B-",
                "C",
                "D",
            ]
            try:
                return 100 - order.index(tok)
            except ValueError:
                return 0

        # Find all [ICRA] tokens
        toks: List[str] = []
        for m in re.finditer(r"\[ICRA\]\s*([A-Za-z0-9+\-()]+)", text, flags=re.I):
            t = re.sub(r"\s+", "", (m.group(1) or "")).upper()
            # strip common suffix like (STABLE) from rating token for LT
            t0 = re.sub(r"\(.*?\)", "", t).strip()
            if t0:
                toks.append(t0)

        # Determine LT and ST
        long_term = ""
        for t in toks:
            if t in lt_vocab and (not long_term or _lt_rank(t) > _lt_rank(long_term)):
                long_term = t
        short_term = parse_rating_tokens_simple(text).get("short_term_rating", "")

        # Outlook from (Stable/Positive/Negative/Developing) if present
        outlook = ""
        mo = OUTLOOK_RE.search(text)
        if mo:
            outlook = (mo.group(1) or "").title()

        updated_on = parse_date(text) or ""
        status = "ok" if (long_term or short_term) else "parsed_no_rating"
        return {
            "agency": "ICRA",
            "company_name": company_name or "",
            "company_id": rid,
            "long_term_rating": long_term,
            "short_term_rating": short_term,
            "outlook": outlook,
            "updated_on": updated_on,
            "source_url": pdf_url,
            "date_retrieved": retrieved_at,
            "status": status,
        }
    except Exception as e:
        return {
            "agency": "ICRA",
            "company_name": company_name or "",
            "company_id": rid,
            "long_term_rating": "",
            "short_term_rating": "",
            "outlook": "",
            "updated_on": "",
            "source_url": pdf_url,
            "date_retrieved": retrieved_at,
            "status": f"error:{type(e).__name__}",
        }


INDIARATINGS_BASE = "https://www.indiaratings.co.in"

INDR_LT_RE = re.compile(
    # Avoid \b boundaries because tokens like "A+" end with '+' (non-word char) and we'd match just "A".
    r"(?<![A-Z0-9])IND\s+(?P<rating>(?:AAA|AA\s*\+|AA\s*-|AA|A\s*\+|A\s*-|A|BBB\s*\+|BBB\s*-|BBB|BB\s*\+|BB\s*-|BB|B\s*\+|B\s*-|B|C|D))(?![A-Z0-9])",
    re.IGNORECASE,
)
INDR_ST_RE = re.compile(
    r"(?<![A-Z0-9])IND\s+(?P<rating>(?:A1\s*\+|A1|A2\s*\+|A2|A3\s*\+|A3|A4\s*\+|A4|A5))(?![A-Z0-9])",
    re.IGNORECASE,
)
INDR_OUTLOOK_RE = re.compile(r"\bOutlook\b[:\\s-]*\\b(Stable|Positive|Negative|Developing)\\b", re.IGNORECASE)


def parse_indiaratings_from_text(text: str) -> Dict[str, str]:
    """
    Extract IND-Ra long-term, short-term, and outlook from press release text.
    """
    txt = (text or "").replace("\u00a0", " ")
    lts: List[str] = []
    sts: List[str] = []
    outlook = ""

    for m in INDR_LT_RE.finditer(txt):
        r = re.sub(r"\s+", "", (m.group("rating") or "")).upper()
        if r:
            lts.append(r)

    for m in INDR_ST_RE.finditer(txt):
        r = re.sub(r"\s+", "", (m.group("rating") or "")).upper()
        if r:
            sts.append(r)

    mo = INDR_OUTLOOK_RE.search(txt)
    if mo:
        outlook = (mo.group(1) or "").title()
    else:
        mo2 = OUTLOOK_RE.search(txt)
        if mo2:
            outlook = (mo2.group(1) or "").title()

    # Best LT rank: AAA highest; D lowest
    lt_order = [
        "AAA",
        "AA+",
        "AA",
        "AA-",
        "A+",
        "A",
        "A-",
        "BBB+",
        "BBB",
        "BBB-",
        "BB+",
        "BB",
        "BB-",
        "B+",
        "B",
        "B-",
        "C",
        "D",
    ]
    lt_rank = {v: i for i, v in enumerate(lt_order)}
    best_lt = ""
    if lts:
        # de-dupe
        uniq = []
        seen = set()
        for x in lts:
            if x in seen:
                continue
            seen.add(x)
            uniq.append(x)
        best_lt = sorted(uniq, key=lambda r: lt_rank.get(r, 10_000))[0]

    st_order = ["A1+", "A1", "A2+", "A2", "A3+", "A3", "A4+", "A4", "A5"]
    st_rank = {v: i for i, v in enumerate(st_order)}
    best_st = ""
    if sts:
        uniq = []
        seen = set()
        for x in sts:
            if x in seen:
                continue
            seen.add(x)
            uniq.append(x)
        best_st = sorted(uniq, key=lambda r: st_rank.get(r, 10_000))[0]

    return {"long_term_rating": best_lt, "short_term_rating": best_st, "outlook": outlook}


def indiaratings_pressrelease_from_id(
    *,
    session: "requests.Session",
    pressrelease_id: str,
    company_name: str = "",
) -> Dict[str, str]:
    """
    Fetch and parse an India Ratings press release.
    Example page:
      https://www.indiaratings.co.in/pressrelease/71503
    Backing JSON:
      https://www.indiaratings.co.in/pressReleases/GetPressreleaseData_BeforeLogin?pressReleaseId=71503
    """
    pid = str(pressrelease_id).strip()
    retrieved_at = dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    source_page = f"{INDIARATINGS_BASE}/pressrelease/{pid}"
    api_url = f"{INDIARATINGS_BASE}/pressReleases/GetPressreleaseData_BeforeLogin"
    try:
        r = session.get(api_url, params={"pressReleaseId": pid}, timeout=30)
        r.raise_for_status()
        items = r.json()
        if not isinstance(items, list) or not items:
            return {
                "agency": "India Ratings",
                "company_name": company_name or "",
                "long_term_rating": "",
                "short_term_rating": "",
                "outlook": "",
                "updated_on": "",
                "source_url": source_page,
                "date_retrieved": retrieved_at,
                "status": "not_found",
            }
        it = items[0] if isinstance(items[0], dict) else {}
        title = str(it.get("pressReleaseTitle") or "")
        effective = str(it.get("effectiveDate") or "")
        overview = str(it.get("overview") or "")
        key = str(it.get("keyRatingDrivers") or "")
        blob = " ".join([title, effective, overview, key])
        pr = parse_indiaratings_from_text(blob)
        # For India Ratings, prefer the explicit effectiveDate field (and later, pressreleasedate from the table).
        # Avoid generic parse_date(blob) because the HTML content often contains many unrelated dates (e.g., today's date).
        updated_on = _normalize_date(effective) or ""

        # Enrich with instrument table (bank facilities) which often contains ST ratings
        # even when the press release narrative doesn't.
        try:
            bf_url = f"{INDIARATINGS_BASE}/pressReleases/GetBankFacilityDataRatingLetter"
            bf = session.get(bf_url, params={"pressReleaseId": pid}, timeout=30)
            bf.raise_for_status()
            bf_items = bf.json()
            rating_blob = ""
            if isinstance(bf_items, list):
                for top in bf_items:
                    if not isinstance(top, dict):
                        continue
                    bfl = top.get("bankFacilitiesList") or []
                    if isinstance(bfl, list):
                        for row in bfl:
                            if not isinstance(row, dict):
                                continue
                            # typical keys include rating / ratingLetter / ratingAction
                            for k in ("rating", "ratingLetter", "ratingAction", "ratingSymbol", "ratingAssigned"):
                                v = row.get(k)
                                if isinstance(v, str) and v.strip():
                                    rating_blob += " " + v.strip()
            if rating_blob.strip():
                pr2 = parse_indiaratings_from_text(rating_blob)
                # Prefer table-derived ratings when available
                pr["long_term_rating"] = pr2.get("long_term_rating") or pr.get("long_term_rating", "")
                pr["short_term_rating"] = pr2.get("short_term_rating") or pr.get("short_term_rating", "")
                # pressreleasedate sometimes present; use it if we don't already have updated_on
                if not updated_on and isinstance(bf_items, list) and bf_items and isinstance(bf_items[0], dict):
                    updated_on = _normalize_date(str(bf_items[0].get("pressreleasedate") or "")) or updated_on
        except Exception:
            pass

        return {
            "agency": "India Ratings",
            "company_name": company_name or "",
            "company_id": "",
            "long_term_rating": pr.get("long_term_rating") or "",
            "short_term_rating": pr.get("short_term_rating") or "",
            "outlook": pr.get("outlook") or "",
            "updated_on": updated_on,
            "source_url": source_page,
            "date_retrieved": retrieved_at,
            "status": "ok" if (pr.get("long_term_rating") or pr.get("short_term_rating")) else "parsed_no_signal",
        }
    except Exception as e:
        return {
            "agency": "India Ratings",
            "company_name": company_name or "",
            "company_id": "",
            "long_term_rating": "",
            "short_term_rating": "",
            "outlook": "",
            "updated_on": "",
            "source_url": source_page,
            "date_retrieved": retrieved_at,
            "status": f"error:{type(e).__name__}",
        }


def indiaratings_search_pressreleases(
    *,
    session: "requests.Session",
    term: str,
    no_of_show_entry: int = 10,
) -> List[Dict[str, str]]:
    """
    Search press releases on India Ratings (returns PR items containing issuer info).
    Backed by:
      /home/GetSearchIssuerData_PressRelease?searchKey=<term>&noOfShowEntry=10
    """
    q = (term or "").strip()
    if not q:
        return []
    url = f"{INDIARATINGS_BASE}/home/GetSearchIssuerData_PressRelease"
    r = session.get(url, params={"searchKey": q, "noOfShowEntry": int(no_of_show_entry)}, timeout=30)
    r.raise_for_status()
    try:
        j = r.json()
    except Exception:
        return []
    if isinstance(j, list):
        return [x for x in j if isinstance(x, dict)]
    if isinstance(j, dict) and isinstance(j.get("data"), list):
        return [x for x in j["data"] if isinstance(x, dict)]
    return []


def indiaratings_pick_best_pressrelease(term: str, hits: Sequence[Dict[str, str]]) -> Optional[Dict[str, str]]:
    """
    Pick best press release hit by issuerName token overlap, breaking ties by latest effectiveDate.
    """
    if not hits:
        return None
    want = _norm_company_name_relaxed(term)

    def _issuer_name(it: Dict[str, str]) -> str:
        # Many responses include issuerName, but some queries return it as null; fall back to title.
        nm = str(it.get("issuerName") or it.get("IssuerName") or it.get("companyName") or it.get("CompanyName") or "").strip()
        if nm:
            return nm
        title = str(it.get("pressReleaseTitle") or it.get("PressReleaseTitle") or "").strip()
        # Heuristic: issuer name is usually before the first colon.
        if ":" in title:
            return title.split(":", 1)[0].strip()
        return title

    def _pr_id(it: Dict[str, str]) -> str:
        return str(it.get("pressReleaseID") or it.get("pressReleaseId") or it.get("PressReleaseID") or it.get("PressReleaseId") or "").strip()

    def _issuer_id(it: Dict[str, str]) -> str:
        return str(it.get("issuerID") or it.get("issuerId") or it.get("IssuerID") or it.get("IssuerId") or "").strip()

    def _effective_iso(it: Dict[str, str]) -> str:
        return _normalize_date(str(it.get("effectiveDate") or it.get("EffectiveDate") or "")) or ""

    best = None
    best_score = -1.0
    best_date = ""
    for it in hits:
        nm = _issuer_name(it)
        pid = _pr_id(it)
        if not nm or not pid:
            continue
        got = _norm_company_name_relaxed(nm)
        wt = set([t for t in want.split() if t])
        gt = set([t for t in got.split() if t])
        if not wt or not gt:
            continue
        inter = len(wt & gt)
        union = len(wt | gt)
        score = (inter / union) if union else 0.0
        if got == want:
            score += 1.0
        iso = _effective_iso(it)
        if score > best_score or (score == best_score and iso and iso > best_date):
            best_score = score
            best_date = iso or best_date
            best = {
                "pressrelease_id": pid,
                "issuer_id": _issuer_id(it),
                "issuer_name": nm,
                "effectiveDate": str(it.get("effectiveDate") or ""),
            }
    return best


def indiaratings_ratings_for_companies(
    *,
    companies: Sequence[str],
    session: "requests.Session",
    sleep_s: float = 0.25,
    max_hits: int = 10,
) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for c in [x.strip() for x in companies if (x or "").strip()]:
        retrieved_at = dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        try:
            # IndiaRatings search endpoint appears to work best with single tokens; multi-word queries often return 0.
            tokens = re.findall(r"[A-Za-z0-9]+", c)
            drop = {"LIMITED", "LTD", "PRIVATE", "PVT", "INDIA", "COMPANY", "CO"}
            variants: List[str] = []
            for t in tokens:
                if len(t) < 4:
                    continue
                if t.upper() in drop:
                    continue
                variants.append(t)
            # Always include the first token if present (even if short), as some companies are acronyms.
            if tokens:
                variants.insert(0, tokens[0])
            # de-dupe preserving order
            seen = set()
            variants = [v for v in variants if v and not (v.lower() in seen or seen.add(v.lower()))]
            variants = variants[:5]  # cap network calls per company

            all_hits: List[Dict[str, str]] = []
            for q in variants:
                hs = indiaratings_search_pressreleases(session=session, term=q, no_of_show_entry=int(max_hits))
                all_hits.extend(hs)
            best = indiaratings_pick_best_pressrelease(c, all_hits)
            if not best:
                out.append(
                    {
                        "agency": "India Ratings",
                        "company_name": c,
                        "long_term_rating": "",
                        "short_term_rating": "",
                        "outlook": "",
                        "updated_on": "",
                        "source_url": "",
                        "date_retrieved": retrieved_at,
                        "status": "not_found",
                    }
                )
                time.sleep(max(0.0, float(sleep_s)))
                continue

            row = indiaratings_pressrelease_from_id(
                session=session,
                pressrelease_id=str(best["pressrelease_id"]),
                company_name=c,
            )
            row["company_id"] = row.get("company_id") or str(best.get("issuer_id") or "")
            out.append(row)
        except Exception as e:
            out.append(
                {
                    "agency": "India Ratings",
                    "company_name": c,
                    "long_term_rating": "",
                    "short_term_rating": "",
                    "outlook": "",
                    "updated_on": "",
                    "source_url": "",
                    "date_retrieved": retrieved_at,
                    "status": f"error:{type(e).__name__}",
                }
            )
        time.sleep(max(0.0, float(sleep_s)))
    return out

CARE_PR_URL_RE = re.compile(
    r"^https?://(?:www\\.)?careratings\\.com/upload/CompanyFiles/PR/(?P<stamp>\\d{12,14})_(?P<name>[^/?#]+)\\.pdf(?:$|[?#])",
    re.IGNORECASE,
)


def _care_score_pr_url(url: str) -> Tuple[int, str]:
    """
    Score CARE PR URLs so we can pick the most recent-looking one.
    Returns (score, url) where higher score is better.
    """
    m = CARE_PR_URL_RE.match((url or "").strip())
    if not m:
        return (0, url)
    stamp = m.group("stamp") or ""
    try:
        # Some URLs use 12 digits (YYYYMMDDHHMM). Some may use 14 (YYYYMMDDHHMMSS).
        return (int(stamp), url)
    except Exception:
        return (0, url)


def care_find_latest_pr_pdf_url(
    *,
    session: "requests.Session",
    company_name: str,
    max_search_results: int = 10,
) -> Optional[str]:
    """
    Find the latest CARE PR PDF for a company by searching for:
      site:careratings.com/upload/CompanyFiles/PR <company>
    """
    def _best_company_hit(hits: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
        # Best-effort name match against CARE's CompanyName
        want = _norm_company_name_relaxed(company_name)
        best = None
        best_score = -1.0
        for h in hits:
            nm = (h.get("CompanyName") or h.get("companyName") or "").strip()
            cid = str(h.get("CompanyID") or h.get("companyId") or "").strip()
            if not nm or not cid:
                continue
            got = _norm_company_name_relaxed(nm)
            # token Jaccard score
            wt = set([t for t in want.split() if t])
            gt = set([t for t in got.split() if t])
            if not wt or not gt:
                continue
            inter = len(wt & gt)
            union = len(wt | gt)
            score = (inter / union) if union else 0.0
            # boost for exact relaxed equality
            if got == want:
                score += 1.0
            if score > best_score:
                best_score = score
                best = {"CompanyName": nm, "CompanyID": cid}
        return best

    # Primary approach: use CARE's own header autocomplete endpoint to get CompanyID,
    # then scrape the company search page for PR PDF links.
    try:
        term = (company_name or "").strip()
        if len(term) >= 4:
            r = session.get(
                "https://www.careratings.com/header/searchlist",
                params={"cinput": term},
                timeout=20,
            )
            r.raise_for_status()
            j = r.json() if r.text else {}
            hits = (j.get("data") or []) if isinstance(j, dict) else []
            if isinstance(hits, list) and hits:
                best = _best_company_hit(hits)
                if best and best.get("CompanyID"):
                    cid = best["CompanyID"]
                    html = session.get(
                        "https://www.careratings.com/search",
                        params={"Id": cid},
                        timeout=25,
                    ).text
                    pr_urls = re.findall(
                        r"https?://www\.careratings\.com/upload/CompanyFiles/PR/[A-Za-z0-9%._-]+\.pdf",
                        html,
                        flags=re.IGNORECASE,
                    )
                    if pr_urls:
                        pr_urls_sorted = sorted(
                            pr_urls, key=lambda u: _care_score_pr_url(u)[0], reverse=True
                        )
                        return pr_urls_sorted[0]
    except Exception:
        pass

    # Fallback: DuckDuckGo search (may be blocked in some networks)
    try:
        q = f"site:careratings.com/upload/CompanyFiles/PR {company_name} pdf"
        urls = duckduckgo_search(q, max_results=max_search_results, session=session)
        pr_urls = [u for u in urls if CARE_PR_URL_RE.match(u or "")]
        if pr_urls:
            pr_urls_sorted = sorted(pr_urls, key=lambda u: _care_score_pr_url(u)[0], reverse=True)
            return pr_urls_sorted[0]
    except Exception:
        pass

    return None


def care_ratings_for_companies(
    *,
    companies: Sequence[str],
    session: "requests.Session",
    sleep_s: float = 0.25,
    max_search_results: int = 10,
) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for c in [x.strip() for x in companies if (x or "").strip()]:
        retrieved_at = dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        # 1) Resolve CARE "companyName" identifier (encoded CompanyID used by their APIs)
        pr_url = ""
        updated_on = ""
        long_term = ""
        short_term = ""
        outlook = ""
        status = "no_company_found"

        # Use the header search endpoint to get CompanyID token
        company_id_token: Optional[str] = None
        try:
            # Try a few query variants to improve hit rate (CARE suggests after 4 chars).
            variants: List[str] = []
            base = (c or "").strip()
            if base:
                variants.append(base)
                # Expand common abbreviations in user-provided names (improves CARE autocomplete hits)
                expanded = base
                expanded = re.sub(r"\bInds\b", "Industries", expanded, flags=re.IGNORECASE)
                expanded = re.sub(r"\bEngg\b", "Engineering", expanded, flags=re.IGNORECASE)
                expanded = expanded.replace("&", "and")
                expanded = re.sub(r"\bCo\b", "Company", expanded, flags=re.IGNORECASE)
                variants.append(expanded)
                # Common legal suffix variations
                variants.append(base + " Ltd")
                variants.append(base + " Limited")
                variants.append(expanded + " Ltd")
                variants.append(expanded + " Limited")
                # Punctuation-stripped variant often helps for names like "S Chand & Co"
                variants.append(re.sub(r"[^A-Za-z0-9 ]+", " ", expanded).strip())
                variants.append(re.sub(r"[^A-Za-z0-9 ]+", " ", expanded).strip() + " Limited")
            # de-dupe preserving order
            seen_v = set()
            variants = [v for v in variants if v and not (v in seen_v or seen_v.add(v))]

            want = _norm_company_name_relaxed(c)
            best = None
            best_score = -1.0
            for term in variants:
                term = term if len(term) >= 4 else (term + " " * (4 - len(term)))
                r = session.get(
                    "https://www.careratings.com/header/searchlist",
                    params={"cinput": term},
                    timeout=20,
                )
                r.raise_for_status()
                j = r.json() if r.text else {}
                hits = (j.get("data") or []) if isinstance(j, dict) else []
                if not (isinstance(hits, list) and hits):
                    continue
                for h in hits:
                    nm = (h.get("CompanyName") or "").strip()
                    cid = str(h.get("CompanyID") or "").strip()
                    if not nm or not cid:
                        continue
                    got = _norm_company_name_relaxed(nm)
                    wt = set([t for t in want.split() if t])
                    gt = set([t for t in got.split() if t])
                    if not wt or not gt:
                        continue
                    inter = len(wt & gt)
                    union = len(wt | gt)
                    score = (inter / union) if union else 0.0
                    if got == want:
                        score += 1.0
                    if score > best_score:
                        best_score = score
                        best = cid
                if best_score >= 1.0:
                    break
            company_id_token = best
        except Exception:
            company_id_token = None

        if not company_id_token:
            # fallback to URL discovery
            pr_url = care_find_latest_pr_pdf_url(
                session=session,
                company_name=c,
                max_search_results=int(max_search_results),
            ) or ""
            if pr_url:
                out.append(care_ratings_from_url(session=session, url=pr_url, company_name=c))
                time.sleep(max(0.0, float(sleep_s)))
                continue
            out.append(
                {
                    "agency": "CARE Ratings",
                    "company_name": c,
                    "long_term_rating": "",
                    "short_term_rating": "",
                    "outlook": "",
                    "updated_on": "",
                    "source_url": "",
                    "date_retrieved": retrieved_at,
                    "status": status,
                }
            )
            time.sleep(max(0.0, float(sleep_s)))
            continue

        # 2) Fetch instrument ratings (no PDF needed)
        numeric_company_id = ""
        try:
            r = session.get(
                "https://www.careratings.com/getSearchprintrating",
                params={"companyName": company_id_token, "YearID": ""},
                timeout=25,
            )
            r.raise_for_status()
            j = r.json() if r.text else {}
            data = (j.get("data") or []) if isinstance(j, dict) else []
            ratings_blob = ""
            if isinstance(data, list) and data:
                try:
                    numeric_company_id = str(data[0].get("CompanyID") or "").strip() if isinstance(data[0], dict) else ""
                except Exception:
                    numeric_company_id = ""
                inst = data[0].get("CompanyInstrument") or []
                if isinstance(inst, list):
                    ratings_blob = " ".join([(x.get("Rating") or "") for x in inst if isinstance(x, dict)])
            pr = parse_care_ratings(ratings_blob)
            long_term = pr.get("long_term_rating") or ""
            short_term = pr.get("short_term_rating") or ""
            outlook = pr.get("outlook") or ""
            status = "ok" if (long_term or short_term) else "parsed_no_signal"
        except Exception as e:
            status = f"error_printrating:{type(e).__name__}"

        # 3) Try to find a PR PDF link + updated_on via getSearchprdocument (if available)
        try:
            r0 = session.get(
                "https://www.careratings.com/getSearchprdocument",
                params={"companyName": company_id_token, "YearID": ""},
                timeout=25,
            )
            r0.raise_for_status()
            j0 = r0.json() if r0.text else {}
            years = (j0.get("dataYear") or []) if isinstance(j0, dict) else []
            year_vals: List[str] = []
            for y in years:
                if isinstance(y, dict):
                    v = str(y.get("PublishedDate") or "").strip()
                    if v.isdigit():
                        year_vals.append(v)
            year_vals_sorted = sorted(set(year_vals), reverse=True)
            # Try years from newest to oldest until we find at least one PRDocument entry
            for y in year_vals_sorted:
                r1 = session.get(
                    "https://www.careratings.com/getSearchprdocument",
                    params={"companyName": company_id_token, "YearID": y},
                    timeout=25,
                )
                r1.raise_for_status()
                j1 = r1.json() if r1.text else {}
                data = (j1.get("data") or []) if isinstance(j1, dict) else []
                if not (isinstance(data, list) and data and isinstance(data[0], dict)):
                    continue
                prdocs = data[0].get("PRDocument") or []
                if not (isinstance(prdocs, list) and prdocs):
                    continue
                # pick latest PublishedDate
                best_doc = None
                best_dt = ""
                for d in prdocs:
                    if not isinstance(d, dict):
                        continue
                    pub = str(d.get("PublishedDate") or "").strip()  # 'YYYY-MM-DD ...'
                    fileurl = str(d.get("FileURL") or "").strip()
                    if not fileurl:
                        continue
                    iso = _normalize_date(pub) or ""
                    if iso and iso > best_dt:
                        best_dt = iso
                        best_doc = fileurl
                if best_doc:
                    pr_url = f"https://www.careratings.com/upload/CompanyFiles/PR/{urllib.parse.quote(best_doc)}"
                    updated_on = best_dt
                    break
        except Exception:
            pass

        out.append(
            {
                "agency": "CARE Ratings",
                "company_name": c,
                "company_id": numeric_company_id or company_id_token,
                "long_term_rating": long_term,
                "short_term_rating": short_term,
                "outlook": outlook,
                "updated_on": updated_on,
                "source_url": pr_url,
                "date_retrieved": retrieved_at,
                "status": status if status else "ok",
            }
        )
        time.sleep(max(0.0, float(sleep_s)))
    return out


ACUITE_LT_RE = re.compile(
    r"\bACUITE\s+(?P<rating>(?:AAA|AA\s*\+|AA\s*-|AA|A\s*\+|A\s*-|A|BBB\s*\+|BBB\s*-|BBB|BB\s*\+|BB\s*-|BB|B\s*\+|B\s*-|B|C|D))",
    re.IGNORECASE,
)
ACUITE_ST_RE = re.compile(
    r"(?<![A-Z0-9])ACUITE\s+(?P<rating>(?:A1\s*\+|A1|A2\s*\+|A2|A3\s*\+|A3|A4\s*\+|A4|A5))(?![A-Z0-9])",
    re.IGNORECASE,
)


def parse_acuite_ratings(text: str) -> Dict[str, str]:
    txt = (text or "").replace("\u00a0", " ")
    lts: List[str] = []
    sts: List[str] = []
    for m in ACUITE_LT_RE.finditer(txt):
        r = re.sub(r"\s+", "", (m.group("rating") or "")).upper()
        if r:
            lts.append(r)
    for m in ACUITE_ST_RE.finditer(txt):
        r = re.sub(r"\s+", "", (m.group("rating") or "")).upper()
        if r:
            sts.append(r)
    # pick best LT
    lt_order = [
        "AAA",
        "AA+",
        "AA",
        "AA-",
        "A+",
        "A",
        "A-",
        "BBB+",
        "BBB",
        "BBB-",
        "BB+",
        "BB",
        "BB-",
        "B+",
        "B",
        "B-",
        "C",
        "D",
    ]
    lt_rank = {v: i for i, v in enumerate(lt_order)}
    best_lt = ""
    if lts:
        seen = set()
        uniq = []
        for x in lts:
            if x in seen:
                continue
            seen.add(x)
            uniq.append(x)
        best_lt = sorted(uniq, key=lambda r: lt_rank.get(r, 10_000))[0]
    st_order = ["A1+", "A1", "A2+", "A2", "A3+", "A3", "A4+", "A4", "A5"]
    st_rank = {v: i for i, v in enumerate(st_order)}
    best_st = ""
    if sts:
        seen = set()
        uniq = []
        for x in sts:
            if x in seen:
                continue
            seen.add(x)
            uniq.append(x)
        best_st = sorted(uniq, key=lambda r: st_rank.get(r, 10_000))[0]
    outlook = ""
    mo = OUTLOOK_RE.search(txt)
    if mo:
        outlook = (mo.group(1) or "").title()
    return {"long_term_rating": best_lt, "short_term_rating": best_st, "outlook": outlook}


def acuite_from_url(
    *,
    session: "requests.Session",
    url: str,
    company_name: str = "",
) -> Dict[str, str]:
    """
    Parse an Acuité press release page like:
      https://connect.acuite.in/fcompany-details/APOLLO_MICRO_SYSTEMS_LIMITED/15th_Jul_25
    """
    retrieved_at = dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    try:
        html = fetch_text(url, session=session, timeout_s=30)
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(html, "html.parser")
        txt = soup.get_text(" ", strip=True)

        # updated_on: usually appears as "Press Release July 15, 2025"
        updated_on = parse_date(txt) or ""

        # Prefer the table near "Product / Quantum / Long Term Rating / Short Term Rating"
        long_term = ""
        short_term = ""
        outlook = ""
        for table in soup.find_all("table"):
            head = " ".join(th.get_text(" ", strip=True) for th in table.find_all(["th"]))
            if "Long Term Rating" not in head and "Short Term Rating" not in head:
                continue
            # collect cell text
            cells = " ".join(td.get_text(" ", strip=True) for td in table.find_all(["td", "th"]))
            pr = parse_acuite_ratings(cells)
            # Outlook is often in the LT cell ("ACUITE A- | Stable")
            long_term = pr.get("long_term_rating") or long_term
            short_term = pr.get("short_term_rating") or short_term
            outlook = pr.get("outlook") or outlook
            if long_term or short_term:
                break

        # Fallback: parse from full page text
        if not long_term and not short_term:
            pr = parse_acuite_ratings(txt)
            long_term = pr.get("long_term_rating") or ""
            short_term = pr.get("short_term_rating") or ""
            outlook = pr.get("outlook") or ""

        status = "ok" if (long_term or short_term) else "parsed_no_signal"
        return {
            "agency": "Acuite",
            "company_name": company_name or "",
            "company_id": "",
            "long_term_rating": long_term,
            "short_term_rating": short_term,
            "outlook": outlook,
            "updated_on": updated_on,
            "source_url": url,
            "date_retrieved": retrieved_at,
            "status": status,
        }
    except Exception as e:
        return {
            "agency": "Acuite",
            "company_name": company_name or "",
            "company_id": "",
            "long_term_rating": "",
            "short_term_rating": "",
            "outlook": "",
            "updated_on": "",
            "source_url": url,
            "date_retrieved": retrieved_at,
            "status": f"error:{type(e).__name__}",
        }


ACUITE_DETAILS_URL_RE = re.compile(
    r"^https?://connect\.acuite\.in/fcompany-details/(?P<slug>[A-Z0-9_]+)/(?P<datepart>[^/?#]+)$",
    re.IGNORECASE,
)


def _acuite_parse_datepart(datepart: str) -> Optional[str]:
    """
    Parse Acuité URL date parts like '15th_Jul_25' -> '2025-07-15'.
    """
    s = (datepart or "").strip()
    m = re.match(r"^(?P<day>\d{1,2})(?:st|nd|rd|th)_(?P<mon>[A-Za-z]{3})_(?P<yy>\d{2})$", s, flags=re.I)
    if not m:
        return None
    day = int(m.group("day"))
    mon = (m.group("mon") or "").title()
    yy = int(m.group("yy"))
    year = 2000 + yy if yy < 80 else 1900 + yy
    mons = {
        "Jan": 1,
        "Feb": 2,
        "Mar": 3,
        "Apr": 4,
        "May": 5,
        "Jun": 6,
        "Jul": 7,
        "Aug": 8,
        "Sep": 9,
        "Oct": 10,
        "Nov": 11,
        "Dec": 12,
    }
    mm = mons.get(mon)
    if not mm:
        return None
    try:
        d = dt.date(year, mm, day)
        return d.isoformat()
    except Exception:
        return None


def _acuite_parse_human_date(s: str) -> Optional[str]:
    """
    Parse Acuité human dates like '5th Sep 25' -> '2025-09-05'.
    """
    t = (s or "").strip()
    m = re.match(r"^(?P<day>\d{1,2})(?:st|nd|rd|th)\s+(?P<mon>[A-Za-z]{3})\s+(?P<yy>\d{2})$", t, flags=re.I)
    if not m:
        return None
    day = int(m.group("day"))
    mon = (m.group("mon") or "").title()
    yy = int(m.group("yy"))
    year = 2000 + yy if yy < 80 else 1900 + yy
    mons = {
        "Jan": 1,
        "Feb": 2,
        "Mar": 3,
        "Apr": 4,
        "May": 5,
        "Jun": 6,
        "Jul": 7,
        "Aug": 8,
        "Sep": 9,
        "Oct": 10,
        "Nov": 11,
        "Dec": 12,
    }
    mm = mons.get(mon)
    if not mm:
        return None
    try:
        return dt.date(year, mm, day).isoformat()
    except Exception:
        return None


def _acuite_iso_to_datepart(iso: str) -> Optional[str]:
    """
    '2025-07-15' -> '15th_Jul_25'
    """
    try:
        y, m, d = iso.split("-")
        day = int(d)
        mon_num = int(m)
        yy = int(y) % 100
    except Exception:
        return None
    mons = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    mon = mons[mon_num - 1] if 1 <= mon_num <= 12 else None
    if not mon:
        return None
    # ordinal suffix
    if 10 <= day % 100 <= 20:
        suf = "th"
    else:
        suf = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return f"{day}{suf}_{mon}_{yy:02d}"


def acuite_autocomplete_companies(
    *,
    session: "requests.Session",
    term: str,
    limit: int = 10,
) -> List[Dict[str, str]]:
    """
    Use Acuité autocomplete endpoint to resolve Company_Code + Company_Name.
    Endpoint: https://connect.acuite.in/search/result?query=<term>
    Returns JSON list of {Company_Name, Company_Code}.
    """
    q = (term or "").strip()
    if not q:
        return []
    r = session.get(
        "https://connect.acuite.in/search/result",
        params={"query": q},
        headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
        timeout=30,
    )
    r.raise_for_status()
    items = r.json()
    out: List[Dict[str, str]] = []
    if isinstance(items, list):
        for it in items[: max(1, int(limit))]:
            if not isinstance(it, dict):
                continue
            nm = str(it.get("Company_Name") or it.get("company_name") or "").strip()
            code = str(it.get("Company_Code") or it.get("company_code") or "").strip()
            if nm and code:
                out.append({"company_name": nm, "company_code": code})
    return out


def acuite_pick_best_company(term: str, hits: Sequence[Dict[str, str]]) -> Optional[Dict[str, str]]:
    if not hits:
        return None
    want = _norm_company_name_relaxed(term)
    best = None
    best_score = -1.0
    for h in hits:
        nm = (h.get("company_name") or "").strip()
        cc = (h.get("company_code") or "").strip()
        if not nm or not cc:
            continue
        got = _norm_company_name_relaxed(nm)
        wt = set([t for t in want.split() if t])
        gt = set([t for t in got.split() if t])
        if not wt or not gt:
            continue
        score = len(wt & gt) / max(1, len(wt | gt))
        if got == want:
            score += 1.0
        if score > best_score:
            best_score = score
            best = {"company_name": nm, "company_code": cc}
    return best


def acuite_latest_update_datepart(
    *,
    session: "requests.Session",
    company_code: str,
) -> Optional[str]:
    """
    Fetch latest instrument modal and extract the latest 'Date of update' to build datepart.
    Endpoint: POST https://connect.acuite.in/latest-intrument
    """
    # CSRF token from liveratings
    html = fetch_text("https://connect.acuite.in/liveratings?page=1", session=session, timeout_s=30)
    m = re.search(r'<meta[^>]+name=\"csrf-token\"[^>]+content=\"([^\"]+)\"', html, flags=re.I)
    token = (m.group(1) if m else "").strip()
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "text/html", "Referer": "https://connect.acuite.in/liveratings?page=1"}
    if token:
        headers["X-CSRF-TOKEN"] = token
        headers["X-Requested-With"] = "XMLHttpRequest"
        headers["Origin"] = "https://connect.acuite.in"
    resp = session.post(
        "https://connect.acuite.in/latest-intrument",
        data={"company_code": str(company_code), "page": "1"},
        headers=headers,
        timeout=30,
    )
    resp.raise_for_status()
    body = resp.text or ""
    # Extract dates like '5th Sep 25'
    dates = re.findall(r"\b\d{1,2}(?:st|nd|rd|th)\s+[A-Za-z]{3}\s+\d{2}\b", body, flags=re.I)
    best_iso = ""
    for d in dates:
        iso = _acuite_parse_human_date(d) or ""
        if iso and iso > best_iso:
            best_iso = iso
    if not best_iso:
        return None
    return _acuite_iso_to_datepart(best_iso)


def _acuite_slug_from_company_name(name: str) -> str:
    s = (name or "").strip()
    if not s:
        return ""
    s = s.upper().replace("&", " AND ")
    # keep parentheses, drop other punctuation
    s = re.sub(r"[^A-Z0-9()]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def acuite_find_latest_company_url(
    *,
    session: "requests.Session",
    company_name: str,
    max_search_results: int = 10,
) -> Optional[str]:
    """
    Find the latest Acuité company-details URL.
    Preferred approach: use Acuité's own live-ratings search endpoint.
    Fallback: DuckDuckGo (may be blocked in some networks).
    """
    # 1) Internal endpoints (best): autocomplete -> latest-intrument date -> construct fcompany-details URL
    try:
        hits = acuite_autocomplete_companies(session=session, term=company_name, limit=10)
        best = acuite_pick_best_company(company_name, hits)
        if best:
            datepart = acuite_latest_update_datepart(session=session, company_code=best["company_code"])
            if datepart:
                slug = _acuite_slug_from_company_name(best["company_name"])
                if slug:
                    url = f"https://connect.acuite.in/fcompany-details/{urllib.parse.quote(slug)}/{urllib.parse.quote(datepart)}"
                    return url
    except Exception:
        pass

    # 2) Fallback: DuckDuckGo discovery
    try:
        q = f"site:connect.acuite.in/fcompany-details {company_name}"
        urls = duckduckgo_search(q, max_results=max_search_results, session=session)
        cand = []
        for u in urls:
            m = ACUITE_DETAILS_URL_RE.match((u or "").strip())
            if not m:
                continue
            iso = _acuite_parse_datepart(m.group("datepart") or "") or ""
            cand.append((iso, u))
        if cand:
            cand.sort(key=lambda t: t[0], reverse=True)
            return cand[0][1]
    except Exception:
        pass

    return None


def _acuite_slug_variants(company_name: str) -> List[str]:
    """
    Generate likely Acuité slugs.
    Example: "Apollo Micro Systems" -> ["APOLLO_MICRO_SYSTEMS", "APOLLO_MICRO_SYSTEMS_LIMITED", ...]
    """
    s = (company_name or "").upper()
    s = s.replace("&", " AND ")
    s = re.sub(r"[^A-Z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return []
    base = s.replace(" ", "_")
    variants = [base]
    # common suffixes
    for suf in ["LIMITED", "LTD", "PRIVATE_LIMITED", "PVT_LTD", "INDIA"]:
        if not base.endswith("_" + suf):
            variants.append(base + "_" + suf)
    # de-dupe preserving order
    seen = set()
    out = []
    for v in variants:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def acuite_resolve_latest_url_by_slug(
    *,
    session: "requests.Session",
    slug: str,
) -> Optional[str]:
    """
    Try to resolve the latest dated URL for a given slug by probing candidate patterns.
    Some undated slug URLs return 500, so we rely on known date patterns only.
    """
    # If user already provides a valid dated URL slug, just return it.
    base = f"https://connect.acuite.in/fcompany-details/{slug}"
    # We don't know latest date programmatically without an index; however, some slugs have a "latest" landing
    # page that returns 500. So we can't derive date purely from slug without discovery.
    return None

def acuite_ratings_for_companies(
    *,
    companies: Sequence[str],
    session: "requests.Session",
    sleep_s: float = 0.25,
    max_search_results: int = 10,
) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for c in [x.strip() for x in companies if (x or "").strip()]:
        url: Optional[str] = None
        company_id = ""
        # 1) Try DuckDuckGo discovery
        try:
            # Prefer internal autocomplete to also capture company_code as company_id
            hits = acuite_autocomplete_companies(session=session, term=c, limit=10)
            best = acuite_pick_best_company(c, hits)
            if best:
                company_id = best.get("company_code") or ""
            url = acuite_find_latest_company_url(session=session, company_name=c, max_search_results=int(max_search_results))
        except Exception:
            url = None
        # 2) If DDG is blocked/empty, we currently cannot deterministically derive the dated URL from name alone.
        # Keep a placeholder for future Acuité internal search integration.
        if not url:
            out.append(
                {
                    "agency": "Acuite",
                    "company_name": c,
                    "company_id": company_id,
                    "long_term_rating": "",
                    "short_term_rating": "",
                    "outlook": "",
                    "updated_on": "",
                    "source_url": "",
                    "date_retrieved": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                    "status": "not_found",
                }
            )
        else:
            row = acuite_from_url(session=session, url=url, company_name=c)
            row["company_id"] = row.get("company_id") or company_id
            out.append(row)
        time.sleep(max(0.0, float(sleep_s)))
    return out

def duckduckgo_search(
    query: str,
    max_results: int = 5,
    session: Optional["requests.Session"] = None,
) -> List[str]:
    """
    DuckDuckGo HTML endpoint returns static HTML results that are easy to parse.
    """
    _require_requests()
    assert requests is not None
    url = "https://html.duckduckgo.com/html/"
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; credit-ratings-scraper/1.0)",
        "Accept": "text/html",
    }
    s = session or requests.Session()
    resp = s.get(url, headers=headers, params={"q": query}, timeout=20)
    resp.raise_for_status()
    html = resp.text

    # Result links are typically <a class="result__a" href="...">
    links = re.findall(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"', html)
    cleaned: List[str] = []
    for href in links:
        if href not in cleaned:
            cleaned.append(href)
        if len(cleaned) >= max_results:
            break
    return cleaned


def _domain(url: str) -> str:
    m = re.match(r"^https?://([^/]+)/", url)
    return (m.group(1) if m else "").lower()


def pick_best_url(urls: Sequence[str], allowed_domains: Sequence[str]) -> Optional[str]:
    allowed = {d.lower() for d in allowed_domains}
    for u in urls:
        dom = _domain(u)
        if any(dom == d or dom.endswith("." + d) for d in allowed):
            return u
    return urls[0] if urls else None


def parse_date(text: str, *, today: Optional[dt.date] = None) -> Optional[str]:
    """
    Tries to parse an "updated on" style date from free text.
    Returns ISO date (YYYY-MM-DD) if found.
    """
    today_iso = (today or dt.date.today()).isoformat()

    # Prefer explicit "update/action/outstanding" dates; ignore forward-looking projections.
    primary_patterns = [
        r"Rating Outstanding as on\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4})",
        r"Date of (?:last )?rating action\s*[:\-]?\s*([A-Za-z]+\s+\d{1,2},?\s+\d{4})",
        r"Rating action date\s*[:\-]?\s*([A-Za-z]+\s+\d{1,2},?\s+\d{4})",
        r"Last updated on\s*[:\-]?\s*([A-Za-z]+\s+\d{1,2},?\s+\d{4})",
        r"Dated\s*[:\-]?\s*([A-Za-z]+\s+\d{1,2},?\s+\d{4})",
    ]

    # Secondary patterns: rationale header dates like "January 02, 2026 | Mumbai".
    secondary_patterns = [
        r"\b([A-Za-z]+\s+\d{1,2},\s+\d{4})\b",
        r"\b([A-Za-z]+\s+\d{1,2}\s+\d{4})\b",
        r"\b([A-Za-z]{3}\s+\d{1,2},\s+\d{4})\b",
        r"\b(\d{1,2}\s+[A-Za-z]{3}\s+\d{2,4})\b",
    ]

    def _best_from(patterns: Sequence[str]) -> str:
        best = ""
        for pat in patterns:
            for m in re.finditer(pat, text, flags=re.IGNORECASE):
                raw = (m.group(1) or "").strip()
                iso = _normalize_date(raw) or ""
                if not iso:
                    continue
                # Reject future dates: documents include projections like "as on March 31, 2026".
                if iso > today_iso:
                    continue
                if not best or iso > best:
                    best = iso
        return best

    best = _best_from(primary_patterns)
    if best:
        return best

    best = _best_from(secondary_patterns)
    return best or None


def _normalize_date(raw: str) -> Optional[str]:
    raw = raw.strip().replace(",", "")
    fmts = [
        "%B %d %Y",  # January 27 2026
        "%b %d %Y",  # Jan 27 2026
        "%d %b %Y",  # 27 Jan 2026
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%d-%m-%y",
        "%d/%m/%y",
    ]
    for fmt in fmts:
        try:
            d = dt.datetime.strptime(raw, fmt).date()
            return d.isoformat()
        except Exception:
            pass
    return None


def _max_iso_date(*dates: str) -> str:
    """
    Return the latest ISO date (YYYY-MM-DD) among the provided strings.
    Non-ISO / empty strings are ignored.
    """
    best = ""
    for d in dates:
        d = (d or "").strip()
        if not d:
            continue
        if re.match(r"^\d{4}-\d{2}-\d{2}$", d):
            if not best or d > best:
                best = d
    return best


def parse_ratings(text: str) -> Dict[str, Optional[str]]:
    """
    Extract internal (long-term) + external (short-term) ratings and outlook, if any.
    This is intentionally heuristic.
    """

    def _first_long_token(s: str) -> Optional[str]:
        for m in LONG_TERM_RATING_RE.finditer(s):
            tok = m.group(1).upper()
            if tok in {"A", "B", "C", "D"}:
                nxt = s[m.end() : m.end() + 1]
                if nxt.isdigit() or nxt == "+":
                    continue
            return tok
        return None

    def _first_short_token(s: str) -> Optional[str]:
        m = SHORT_TERM_RATING_RE.search(s)
        return re.sub(r"\s+", "", m.group(1)).upper() if m else None

    def _find_with_context(
        pattern: "re.Pattern[str]",
        ctx_keywords: Sequence[str],
        window: int = 90,
    ) -> Optional[str]:
        lower = text.lower()
        for m in pattern.finditer(text):
            start, end = m.start(), m.end()
            lo = max(0, start - window)
            hi = min(len(lower), end + window)
            neighborhood = lower[lo:hi]
            if any(k.lower() in neighborhood for k in ctx_keywords):
                token = m.group(1).upper()
                if token in {"A", "B", "C", "D"}:
                    # Avoid false positives where we match the "A" in "A3"/"A3+" etc.
                    prev_ch = text[start - 1 : start] if start > 0 else ""
                    next_ch = text[end : end + 1] if end < len(text) else ""
                    if prev_ch.isdigit() or next_ch.isdigit() or next_ch == "+":
                        continue
                    strong_ctx = ("long", "long-term", "issuer", "lt", "long term")
                    if not any(k in neighborhood for k in strong_ctx):
                        continue
                return token
        return None

    # Prefer a section-based parse: the first "Long Term ..." section up to the first "Short Term ..."
    # This avoids mis-assigning long-term as the "A" inside "A3/A3+" when the document repeats sections.
    seg = re.search(
        r"Long[\s-]?term(?:\s+Rating)?[^A-Za-z0-9]{0,20}(?P<long_sec>.{0,600}?)"
        r"Short[\s-]?term(?:\s+Rating)?[^A-Za-z0-9]{0,20}(?P<short_sec>.{0,600})",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if seg:
        long_rating = _first_long_token(seg.group("long_sec")) or ""
        short_rating = _first_short_token(seg.group("short_sec")) or ""
        # If we got a short-term rating but missed long-term in the segment, try finding long-term in the full text with context.
        if not long_rating and short_rating:
            long_rating = _find_with_context(
                LONG_TERM_RATING_RE,
                ctx_keywords=("long", "long-term", "issuer", "rating", "rated", "lt"),
            ) or ""
        if long_rating or short_rating:
            outlook_m = OUTLOOK_RE.search(text)
            outlook = outlook_m.group(1).capitalize() if outlook_m else None
            return {
                "internal_rating": long_rating or None,
                "external_rating": short_rating or None,
                "outlook": outlook,
            }

    # Fallback mode: only accept a rating token if it appears near rating-specific context.
    long_rating = _find_with_context(
        LONG_TERM_RATING_RE,
        ctx_keywords=("long", "long-term", "issuer", "rating", "rated", "lt"),
    )
    short_rating = _find_with_context(
        SHORT_TERM_RATING_RE,
        ctx_keywords=("short", "short-term", "rating", "rated", "st"),
    )

    outlook_m = OUTLOOK_RE.search(text)
    outlook = outlook_m.group(1).capitalize() if outlook_m else None

    return {
        "internal_rating": long_rating,
        "external_rating": short_rating,
        "outlook": outlook,
    }


def parse_rating_tokens_simple(text: str) -> Dict[str, str]:
    """
    Lightweight parser for rating tokens when we only have a short string like:
      "Crisil AA-/Stable" or "CRISIL A1+"
    (no context words like 'Long-term'/'Short-term').
    """
    text = text or ""
    # Avoid false positives where LONG_TERM_RATING_RE matches "A" in "A3" etc.
    m_long = None
    for m in LONG_TERM_RATING_RE.finditer(text):
        tok = m.group(1).upper()
        if tok in {"A", "B", "C", "D"}:
            nxt = text[m.end() : m.end() + 1]
            if nxt.isdigit():
                continue
        m_long = m
        break

    m_short = SHORT_TERM_RATING_RE.search(text)
    out_m = OUTLOOK_RE.search(text)
    return {
        "long_term_rating": (m_long.group(1).upper() if m_long else ""),
        "short_term_rating": (re.sub(r"\s+", "", m_short.group(1)).upper() if m_short else ""),
        "outlook": (out_m.group(1).capitalize() if out_m else ""),
    }


@dataclass(frozen=True)
class Agency:
    name: str
    query_suffix: str
    allowed_domains: Tuple[str, ...]
    fixed_url: Optional[str] = None
    use_search: bool = True


AGENCIES: Tuple[Agency, ...] = (
    Agency("CRISIL", "CRISIL credit rating", ("crisilratings.com", "crisil.com"), use_search=False),
    Agency("CARE Edge", "CARE Edge credit rating", ("careedge.in", "careratings.com")),
    Agency("ICRA", "ICRA credit rating", ("icra.in",)),
    Agency("India Ratings", "India Ratings credit rating", ("indiaratings.co.in",)),
    Agency("InCred Ratings", "InCred Ratings credit rating", ("incredratings.com",)),
)


CRISIL_BASE = "https://www.crisilratings.com"
CRISIL_DOCS_BASE = "https://www.crisil.com"
CRISIL_CREDIT_RATINGS_RESOURCE = (
    "/content/crisilratings/en/home/our-business/ratings/credit-ratings-list/"
    "_jcr_content/wrapper_100_par/columncontrol_copy/container-100-1/ratingresultlisting"
)
CRISIL_RATING_RATIONALE_RESOURCE = (
    "/content/crisilratings/en/home/our-business/ratings/rating-rationale/"
    "_jcr_content/wrapper_100_par/ratingresultlisting"
)
CRISIL_RATINGDOCS_BASE_PATH = "/mnt/winshare/Ratings/RatingList/RatingDocs/"

ICRA_BASE = "https://www.icra.in"


def _safe_join_url(base: str, path: str) -> str:
    if path.startswith("http://") or path.startswith("https://"):
        return path
    parts = path.split("/")
    enc = "/".join(urllib.parse.quote(p) for p in parts)
    if not enc.startswith("/"):
        enc = "/" + enc
    return base.rstrip("/") + enc


def _crisil_view_rating_url(rating_file_base: str, pr_doc: str) -> str:
    """
    Build the 'View Rating' (rating rationale / bulletin) URL from CRISIL list fields.

    CRISIL returns `ratingFileBasePath` as a path like `/mnt/winshare/.../RatingDocs/`,
    but those documents are hosted on `www.crisil.com` (not `www.crisilratings.com`).
    """
    rating_file_base = (rating_file_base or "").strip()
    pr_doc = (pr_doc or "").strip()
    if not rating_file_base or not pr_doc:
        return ""

    path = rating_file_base.rstrip("/") + "/" + pr_doc
    # If CRISIL gives a relative /mnt/... path, it is often accessible on both hosts.
    # Prefer `www.crisilratings.com` (some docs resolve there even when `www.crisil.com` doesn't),
    # and rely on `fetch_text_crisil_doc()` to swap hosts on 404 when needed.
    base = CRISIL_BASE if path.startswith("/mnt/") else CRISIL_BASE
    return _safe_join_url(base, path)


def _crisil_results_url() -> str:
    return f"{CRISIL_BASE}{CRISIL_CREDIT_RATINGS_RESOURCE}.results.json"

def _crisil_rating_rationale_results_url() -> str:
    return f"{CRISIL_BASE}{CRISIL_RATING_RATIONALE_RESOURCE}.results.json"


def _crisil_default_headers() -> Dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/121.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json",
    }


def fetch_crisil_row(
    company: str,
    session: "requests.Session",
    retrieved_at: str,
    limit: int = 50,
) -> Dict[str, str]:
    """
    Company-specific CRISIL flow:
    - calls the public list endpoint with filters (best-effort)
    - fetches the 'View rating' document and parses ratings/date heuristically
    """
    url = _crisil_results_url()
    params = {
        "cmd": "CR",
        "start": 0,
        "limit": limit,
        "filters": json.dumps({"company_name": company}),
    }
    resp = session.get(url, params=params, headers=_crisil_default_headers(), timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    docs_raw = payload.get("docs")
    if not docs_raw:
        return {
            "company_name": company,
            "agency": "CRISIL",
            "internal_rating": "",
            "external_rating": "",
            "updated_on": "",
            "outlook": "",
            "source_url": "",
            "date_retrieved": retrieved_at,
            "status": "not_found",
        }

    try:
        grouped = json.loads(docs_raw)
    except Exception:
        grouped = {}

    items: List[dict] = []
    if isinstance(grouped, dict):
        for _, arr in grouped.items():
            if isinstance(arr, list):
                items.extend([x for x in arr if isinstance(x, dict)])

    if not items:
        return {
            "company_name": company,
            "agency": "CRISIL",
            "internal_rating": "",
            "external_rating": "",
            "updated_on": "",
            "outlook": "",
            "source_url": "",
            "date_retrieved": retrieved_at,
            "status": "parsed_no_results",
        }

    company_l = company.lower()

    def _score(it: dict) -> int:
        name = str(it.get("companyName", "")).lower()
        if name == company_l:
            return 100
        if company_l and company_l in name:
            return 80
        return 0

    items.sort(key=_score, reverse=True)

    best_status = "parsed_no_rating"
    for it in items[:5]:
        rating_file_base = str(it.get("ratingFileBasePath", "") or "")
        pr_doc = str(it.get("prDocument", "") or "")
        if not rating_file_base or not pr_doc:
            continue

        rationale_url = _crisil_view_rating_url(rating_file_base, pr_doc)
        try:
            rationale_html = fetch_text_crisil_doc(rationale_url, session=session, timeout_s=30)
            rationale_text = html_to_text(rationale_html)
            parsed = parse_ratings(rationale_text)
            updated_on = parse_date(rationale_text) or ""

            fallback_rating = str(it.get("rating", "") or "").strip()

            internal_rating = parsed.get("internal_rating") or ""
            external_rating = parsed.get("external_rating") or ""
            outlook = parsed.get("outlook") or str(it.get("outlook", "") or "")

            if not internal_rating and not external_rating and fallback_rating:
                pr = parse_ratings(fallback_rating)
                internal_rating = pr.get("internal_rating") or ""
                external_rating = pr.get("external_rating") or fallback_rating

            status = "ok" if (internal_rating or external_rating) else "parsed_no_rating"
            best_status = status
            return {
                "company_name": company,
                "agency": "CRISIL",
                "internal_rating": internal_rating,
                "external_rating": external_rating,
                "updated_on": updated_on,
                "outlook": outlook,
                "source_url": rationale_url,
                "date_retrieved": retrieved_at,
                "status": status,
            }
        except Exception as e:
            best_status = f"error: {type(e).__name__}"
            continue

    return {
        "company_name": company,
        "agency": "CRISIL",
        "internal_rating": "",
        "external_rating": "",
        "updated_on": "",
        "outlook": "",
        "source_url": "",
        "date_retrieved": retrieved_at,
        "status": best_status,
    }


def fetch_crisil_credit_ratings_list_page(
    *,
    session: "requests.Session",
    start: int,
    end: int,
    filters: Optional[Dict[str, str]] = None,
) -> Tuple[int, List[Dict[str, str]]]:
    """
    Fetch a slice of the CRISIL "Credit Rating List" table.

    Important: the website uses `limit = limit + start` and sends `start=<offset>` and
    `limit=<offset+page_size>` (limit behaves like an END index, not a page size).

    Returns: (numFound, flattened_rows)
    """
    url = _crisil_results_url()
    params = {
        "cmd": "CR",
        "start": int(start),
        "limit": int(end),
        "filters": json.dumps(filters or {}, ensure_ascii=False),
    }
    resp = session.get(url, params=params, headers=_crisil_default_headers(), timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    num_found = int(payload.get("numFound") or 0)
    docs_raw = payload.get("docs") or ""
    try:
        grouped = json.loads(docs_raw) if isinstance(docs_raw, str) else {}
    except Exception:
        grouped = {}

    rows: List[Dict[str, str]] = []
    if isinstance(grouped, dict):
        for company_code, items in grouped.items():
            if not isinstance(items, list):
                continue
            for it in items:
                if not isinstance(it, dict):
                    continue
                rating_file_base = str(it.get("ratingFileBasePath", "") or "")
                pr_doc = str(it.get("prDocument", "") or "")
                view_rating_url = _crisil_view_rating_url(rating_file_base, pr_doc)
                rows.append(
                    {
                        "company_code": str(it.get("companyCode") or company_code or ""),
                        "company_name": str(it.get("companyName") or ""),
                        "industry_code": str(it.get("industryCode") or ""),
                        "industry_name": str(it.get("industryName") or ""),
                        "instrument_name": str(it.get("instrumentName") or ""),
                        "rating": str(it.get("rating") or ""),
                        "outlook": str(it.get("outlook") or ""),
                        "product": str(it.get("product") or ""),
                        "rating_file_base_path": rating_file_base,
                        "pr_document": pr_doc,
                        "view_rating_url": view_rating_url,
                    }
                )

    return num_found, rows


def scrape_crisil_credit_ratings_list(
    *,
    session: "requests.Session",
    page_size_companies: int = 50,
    filters: Optional[Dict[str, str]] = None,
    max_companies: Optional[int] = None,
    sleep_s: float = 0.25,
) -> List[Dict[str, str]]:
    all_rows: List[Dict[str, str]] = []
    start = 0
    num_found: Optional[int] = None

    while True:
        end = start + int(page_size_companies)
        nf, rows = fetch_crisil_credit_ratings_list_page(
            session=session,
            start=start,
            end=end,
            filters=filters,
        )
        num_found = nf if num_found is None else num_found
        if not rows:
            break
        all_rows.extend(rows)
        if max_companies is not None and end >= int(max_companies):
            break
        if num_found and end >= num_found:
            break
        start = end
        time.sleep(max(0.0, float(sleep_s)))

    return all_rows

def fetch_crisil_rating_rationale_page(
    *,
    session: "requests.Session",
    start: int,
    end: int,
) -> Tuple[int, List[Dict[str, str]]]:
    """
    Fetch a slice of CRISIL's 'Latest Rating Rationales' list (cmd=RR).
    """
    url = _crisil_rating_rationale_results_url()
    params = {
        "cmd": "RR",
        "start": int(start),
        "limit": int(end),
        "filters": json.dumps({}, ensure_ascii=False),
    }
    resp = session.get(url, params=params, headers=_crisil_default_headers(), timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    num_found = int(payload.get("numFound") or 0)
    docs = payload.get("docs") or []
    out: List[Dict[str, str]] = []
    if isinstance(docs, list):
        for d in docs:
            if isinstance(d, dict):
                out.append({str(k): ("" if v is None else str(v)) for k, v in d.items()})
    return num_found, out


def build_crisil_rating_rationale_index(
    *,
    session: "requests.Session",
    step: int = 2000,
    sleep_s: float = 0.05,
) -> Dict[str, Dict[str, str]]:
    """
    Build an index of latest rating-rationale record per company (normalized by strict name).
    """
    idx: Dict[str, Dict[str, str]] = {}
    start = 0
    num_found: Optional[int] = None

    while True:
        end = start + int(step)
        nf, docs = fetch_crisil_rating_rationale_page(session=session, start=start, end=end)
        num_found = nf if num_found is None else num_found
        if not docs:
            break
        for d in docs:
            name = d.get("companyName") or ""
            key = _norm_company_name(name)
            trans = d.get("transDate") or d.get("ratingDate") or ""
            iso = _normalize_date(trans) or ""
            prev = idx.get(key)
            if not prev:
                d["_trans_iso"] = iso
                idx[key] = d
            else:
                prev_iso = prev.get("_trans_iso") or ""
                if iso and (not prev_iso or iso > prev_iso):
                    d["_trans_iso"] = iso
                    idx[key] = d
        if num_found and end >= num_found:
            break
        start = end
        time.sleep(max(0.0, float(sleep_s)))

    return idx


def export_crisil_ratingdocs_urls(
    *,
    session: "requests.Session",
    out_path: str,
    ratingdocs_prefix_url: str = "https://www.crisil.com/mnt/winshare/Ratings/RatingList/RatingDocs/",
    page_size_companies: int = 50,
    max_companies: Optional[int] = None,
    sleep_s: float = 0.25,
) -> int:
    """
    Enumerate CRISIL "View rating" document URLs and write all unique URLs that live under
    the given `ratingdocs_prefix_url` (defaulting to CRISIL's RatingDocs directory).

    Note: CRISIL does not provide a public directory listing for RatingDocs. This method
    enumerates what CRISIL publishes via its public "Credit Rating List" endpoint.
    """
    prefix = (ratingdocs_prefix_url or "").strip()
    if not prefix:
        raise ValueError("ratingdocs_prefix_url must be non-empty")
    if not prefix.endswith("/"):
        prefix += "/"

    rows = scrape_crisil_credit_ratings_list(
        session=session,
        page_size_companies=int(page_size_companies),
        filters=None,
        max_companies=max_companies,
        sleep_s=float(sleep_s),
    )

    urls: List[str] = []
    seen: set[str] = set()
    for r in rows:
        u = str(r.get("view_rating_url") or "").strip()
        if not u:
            continue
        if not u.startswith(prefix):
            continue
        if u in seen:
            continue
        seen.add(u)
        urls.append(u)

    urls.sort()
    with open(out_path, "w", encoding="utf-8") as f:
        for u in urls:
            f.write(u + "\n")

    return len(urls)


def _derive_company_from_ratingdocs_url(url: str) -> str:
    """
    Best-effort: infer company token from a RatingDocs URL like:
      .../RatingDocs/<CompanyToken>_<Month%20DD_%2020YY>_RR_<id>.html
    """
    try:
        last = url.rsplit("/", 1)[-1]
        if last.endswith(".html"):
            last = last[:-5]
        # Company token is typically the first segment before the first underscore.
        # Example: CreativeNewtechLimited_October%2025_%202024_RR_355000
        company_token = last.split("_", 1)[0]
        return urllib.parse.unquote(company_token)
    except Exception:
        return ""


def extract_crisil_short_term_and_updated_on_from_url(
    *,
    session: "requests.Session",
    url: str,
    attempts: int = 1,
) -> Dict[str, str]:
    """
    Fetch a CRISIL RatingDocs HTML and extract:
    - short_term_rating: parse_ratings(...)[external_rating]
    - updated_on: parse_date(...)
    """
    company = _derive_company_from_ratingdocs_url(url)
    try:
        html = fetch_text_with_retries(url, session=session, attempts=int(attempts))
        text = html_to_text(html)
        parsed = parse_ratings(text)
        updated_on = parse_date(text) or ""
        short_term = parsed.get("external_rating") or ""
        status = "ok" if (short_term or updated_on) else "parsed_no_signal"
        return {
            "company": company,
            "url": url,
            "short_term_rating": short_term,
            "updated_on": updated_on,
            "status": status,
        }
    except Exception as e:
        return {
            "company": company,
            "url": url,
            "short_term_rating": "",
            "updated_on": "",
            "status": f"error: {type(e).__name__}",
        }


def parse_crisil_ratingdocs_urls_to_csv(
    *,
    session: "requests.Session",
    urls_file: str,
    out_csv: str,
    workers: int = 8,
    max_urls: Optional[int] = None,
    resume: bool = True,
) -> int:
    """
    Read newline-delimited RatingDocs URLs and write CSV with:
      company,url,short_term_rating,updated_on,status

    If resume=True and out_csv exists, skip already-processed URLs.
    Returns number of rows written (newly processed in this run).
    """
    done: set[str] = set()
    if resume:
        try:
            with open(out_csv, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames and "url" in reader.fieldnames:
                    for row in reader:
                        u = (row.get("url") or "").strip()
                        if u:
                            done.add(u)
        except FileNotFoundError:
            pass

    urls: List[str] = []
    with open(urls_file, "r", encoding="utf-8") as f:
        for line in f:
            u = line.strip()
            if not u:
                continue
            if u in done:
                continue
            urls.append(u)
            if max_urls is not None and len(urls) >= int(max_urls):
                break

    if not urls:
        return 0

    file_exists = False
    try:
        with open(out_csv, "r", encoding="utf-8") as _:
            file_exists = True
    except FileNotFoundError:
        file_exists = False

    out_f = open(out_csv, "a", encoding="utf-8", newline="")
    writer = csv.DictWriter(
        out_f,
        fieldnames=["company", "url", "short_term_rating", "updated_on", "status"],
    )
    if not file_exists:
        writer.writeheader()

    wrote = 0
    try:
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
            futs = [
                ex.submit(
                    extract_crisil_short_term_and_updated_on_from_url,
                    session=session,
                    url=u,
                    attempts=1,
                )
                for u in urls
            ]
            for fut in as_completed(futs):
                row = fut.result()
                writer.writerow(row)
                wrote += 1
                if wrote % 200 == 0:
                    out_f.flush()
    finally:
        out_f.close()

    return wrote


def retry_http_error_rows_in_ratingdocs_csv(
    *,
    session: "requests.Session",
    in_csv: str,
    out_csv: str,
    workers: int = 8,
    attempts: int = 4,
    backoff_s: float = 0.8,
) -> Tuple[int, int, int]:
    """
    Read an existing parsed RatingDocs CSV and retry rows with status == 'error: HTTPError'.
    Writes a refreshed CSV to out_csv.

    Returns: (total_rows, http_error_rows_found, http_error_rows_recovered)
    """
    rows: List[Dict[str, str]] = []
    with open(in_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k: (v or "") for k, v in r.items() if k})

    by_url: Dict[str, Dict[str, str]] = {}
    for r in rows:
        u = (r.get("url") or "").strip()
        if u:
            by_url[u] = r

    targets = [u for u, r in by_url.items() if (r.get("status") or "").strip() == "error: HTTPError"]

    recovered = 0
    if targets:
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
            futs = {
                ex.submit(
                    extract_crisil_short_term_and_updated_on_from_url,
                    session=session,
                    url=u,
                    attempts=int(attempts),
                ): u
                for u in targets
            }
            for fut in as_completed(futs):
                u = futs[fut]
                row = fut.result()
                if (row.get("status") or "").startswith("ok") or (row.get("status") or "") == "parsed_no_signal":
                    recovered += 1
                by_url[u] = row

    # Write out in deterministic order
    out_rows = [by_url[u] for u in sorted(by_url.keys())]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["company", "url", "short_term_rating", "updated_on", "status"],
        )
        writer.writeheader()
        for r in out_rows:
            writer.writerow(
                {
                    "company": r.get("company", ""),
                    "url": r.get("url", ""),
                    "short_term_rating": r.get("short_term_rating", ""),
                    "updated_on": r.get("updated_on", ""),
                    "status": r.get("status", ""),
                }
            )

    return (len(out_rows), len(targets), recovered)


def _norm_company_name(s: str) -> str:
    s = (s or "").upper()
    s = s.replace("&", " AND ")
    s = re.sub(r"[^A-Z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _norm_company_name_relaxed(s: str) -> str:
    s = _norm_company_name(s)
    # Common abbreviations / formatting quirks in user-provided company names
    # - "Inds" -> "Industries"
    # - spaced acronyms like "M M P" -> "MMP"
    s = re.sub(r"\bINDS\b", "INDUSTRIES", s)
    toks = s.split()
    collapsed: List[str] = []
    i = 0
    while i < len(toks):
        if len(toks[i]) == 1:
            j = i
            run: List[str] = []
            while j < len(toks) and len(toks[j]) == 1:
                run.append(toks[j])
                j += 1
            # Collapse runs of 2+ single-letter tokens into an acronym
            if len(run) >= 2:
                collapsed.append("".join(run))
                i = j
                continue
        collapsed.append(toks[i])
        i += 1
    s = " ".join(collapsed)
    s = re.sub(r"\bLIMITED\b", "LTD", s)
    s = re.sub(r"\bCOMPANY\b", "CO", s)
    drop = {
        "LTD",
        "LIMITED",
        "PVT",
        "PRIVATE",
        "CO",
        "CORPORATION",
        "CORP",
        "INDIA",
    }
    parts = [p for p in s.split() if p not in drop]
    return " ".join(parts).strip()


def crisil_ratings_for_companies(
    *,
    companies: Sequence[str],
    session: "requests.Session",
    page_size_companies: int = 50,
    sleep_s: float = 0.25,
    enrich_updated_on: bool = False,
) -> List[Dict[str, str]]:
    """
    Optimized implementation:
    - Instead of scraping the entire CR list (17k+ companies) up front, page through it and stop
      as soon as all requested companies are matched.
    - Only if some companies are still missing, page through Rating Rationales (cmd=RR) and stop
      once the missing companies are found.
    """
    retrieved_at = dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    inputs: List[str] = [c.strip() for c in companies if (c or "").strip()]
    if not inputs:
        return []

    wanted_strict: Dict[str, List[str]] = {}
    wanted_relaxed: Dict[str, List[str]] = {}
    for c in inputs:
        wanted_strict.setdefault(_norm_company_name(c), []).append(c)
        wanted_relaxed.setdefault(_norm_company_name_relaxed(c), []).append(c)

    remaining: set[str] = set(inputs)
    inp_relaxed_key: Dict[str, str] = {c: _norm_company_name_relaxed(c) for c in inputs}
    out: List[Dict[str, str]] = []

    # Caches across all rows
    updated_cache: Dict[str, str] = {}
    rating_cache: Dict[str, Dict[str, str]] = {}

    # Track matches discovered in the credit rating list
    matched_inputs: Dict[str, Dict[str, str]] = {}  # input_company -> match metadata

    def _fuzzy_match_inputs(candidate_relaxed: str) -> List[str]:
        """
        Conservative fuzzy matcher for remaining inputs.
        Only matches when token overlap is high (Jaccard >= 0.70) with at least 2
        meaningful tokens (len>=3). Avoids raw substring checks (e.g., 'INDU' in 'INDUSTRIES').
        """
        def _meaningful_tokens(s: str) -> List[str]:
            toks = [t for t in (s or "").split() if t]
            toks = [t for t in toks if len(t) >= 3]
            # generic suffixes that are not very discriminative
            drop = {"INDUSTRIES", "INDUSTRY", "INDUSTRIAL"}
            return [t for t in toks if t not in drop]

        cand = (candidate_relaxed or "").strip()
        if not cand:
            return []
        cand_toks = _meaningful_tokens(cand)
        if len(cand_toks) < 2:
            return []
        cand_set = set(cand_toks)
        out_inps: List[str] = []
        for inp in list(remaining):
            key = inp_relaxed_key.get(inp, "")
            if not key:
                continue
            inp_toks = _meaningful_tokens(key)
            if len(inp_toks) < 2:
                continue
            inp_set = set(inp_toks)
            inter = len(inp_set & cand_set)
            union = len(inp_set | cand_set)
            if union and (inter / union) >= 0.70:
                out_inps.append(inp)
        return out_inps

    # 1) Page through the Credit Rating List until we match everything or hit the end.
    start = 0
    num_found: Optional[int] = None
    step = max(50, int(page_size_companies))
    while remaining:
        end = start + step
        nf, rows = fetch_crisil_credit_ratings_list_page(session=session, start=start, end=end, filters=None)
        num_found = nf if num_found is None else num_found
        if not rows:
            break

        # Group this page by company_code to compare names once per company.
        by_code: Dict[str, List[Dict[str, str]]] = {}
        for r in rows:
            r["date_retrieved"] = retrieved_at
            code = r.get("company_code") or ""
            if code:
                by_code.setdefault(code, []).append(r)

        for code, items in by_code.items():
            name = (items[0].get("company_name") or "").strip()
            if not name:
                continue
            k_strict = _norm_company_name(name)
            k_relaxed = _norm_company_name_relaxed(name)

            # Strict match takes priority
            candidates = wanted_strict.get(k_strict) or []
            match_type = "strict"
            if not candidates:
                candidates = wanted_relaxed.get(k_relaxed) or []
                match_type = "relaxed"
            if not candidates:
                # Fuzzy fallback for small remaining sets
                candidates = _fuzzy_match_inputs(k_relaxed)
                match_type = "fuzzy"
            if not candidates:
                continue

            for inp in list(candidates):
                if inp not in remaining:
                    continue
                matched_inputs[inp] = {
                    "matched_company_name": name,
                    "match_type": match_type,
                    "company_code": code,
                }
                remaining.discard(inp)
                # Emit all instrument rows for this matched company_code from this page
                for row in items:
                    view_url = row.get("view_rating_url") or ""
                    rating_str = row.get("rating") or ""
                    toks = parse_rating_tokens_simple(rating_str)
                    long_term = toks.get("long_term_rating", "")
                    short_term = toks.get("short_term_rating", "")
                    outlook = (row.get("outlook") or "") or toks.get("outlook", "")
                    updated_on = ""

                    if enrich_updated_on and view_url:
                        if view_url in updated_cache:
                            updated_on = updated_cache[view_url]
                            pr = rating_cache.get(view_url) or {}
                            long_term = pr.get("long_term_rating") or long_term
                            short_term = pr.get("short_term_rating") or short_term
                            outlook = pr.get("outlook") or outlook
                        else:
                            try:
                                html = fetch_text_crisil_doc(view_url, session=session, timeout_s=30)
                                txt = html_to_text(html)
                                updated_on = parse_date(txt) or ""
                                pr_raw = parse_ratings(txt)
                                pr = {
                                    "long_term_rating": pr_raw.get("internal_rating") or "",
                                    "short_term_rating": pr_raw.get("external_rating") or "",
                                    "outlook": pr_raw.get("outlook") or "",
                                }
                                if not pr["long_term_rating"] or not pr["short_term_rating"]:
                                    simple = parse_rating_tokens_simple(txt)
                                    pr["long_term_rating"] = pr["long_term_rating"] or simple.get("long_term_rating", "")
                                    pr["short_term_rating"] = pr["short_term_rating"] or simple.get("short_term_rating", "")
                                    pr["outlook"] = pr["outlook"] or simple.get("outlook", "")
                                updated_cache[view_url] = updated_on
                                rating_cache[view_url] = pr
                                long_term = pr.get("long_term_rating") or long_term
                                short_term = pr.get("short_term_rating") or short_term
                                outlook = pr.get("outlook") or outlook
                            except Exception:
                                updated_on = ""
                                updated_cache[view_url] = ""
                            time.sleep(max(0.0, float(sleep_s)))

                    out.append(
                        {
                            "input_company_name": inp,
                            "matched_company_name": name,
                            "match_type": match_type,
                            "company_code": code,
                            "industry_name": row.get("industry_name") or "",
                            "instrument_name": row.get("instrument_name") or "",
                            "rating": rating_str,
                            "outlook": outlook,
                            "long_term_rating": long_term,
                            "short_term_rating": short_term,
                            "updated_on": updated_on,
                            "view_rating_url": view_url,
                            "date_retrieved": retrieved_at,
                        }
                    )

        if num_found and end >= num_found:
            break
        start = end
        time.sleep(max(0.0, float(sleep_s)))

    # 2) Fallback for any still-missing companies: scan Rating Rationales (cmd=RR) progressively.
    if remaining:
        rr_start = 0
        rr_nf: Optional[int] = None
        rr_step = 5000  # fewer HTTP calls than 2000; endpoint supports large page sizes
        while remaining:
            rr_end = rr_start + rr_step
            nf, docs = fetch_crisil_rating_rationale_page(session=session, start=rr_start, end=rr_end)
            rr_nf = nf if rr_nf is None else rr_nf
            if not docs:
                break
            for d in docs:
                name = (d.get("companyName") or "").strip()
                if not name:
                    continue
                k_strict = _norm_company_name(name)
                k_relaxed = _norm_company_name_relaxed(name)
                candidates = wanted_strict.get(k_strict) or []
                if not candidates:
                    candidates = wanted_relaxed.get(k_relaxed) or []
                match_type = "rationale"
                if not candidates:
                    candidates = _fuzzy_match_inputs(k_relaxed)
                    match_type = "fuzzy_rationale"
                if not candidates:
                    continue

                rating_file = d.get("ratingFileName") or ""
                view_url = _crisil_view_rating_url(CRISIL_RATINGDOCS_BASE_PATH, rating_file)
                rating_str = d.get("heading") or ""
                rr_iso = _normalize_date(d.get("transDate") or d.get("ratingDate") or "") or ""
                updated_on = rr_iso
                toks = parse_rating_tokens_simple(rating_str)
                long_term = toks.get("long_term_rating", "")
                short_term = toks.get("short_term_rating", "")
                outlook = toks.get("outlook", "")

                # Enrich from the doc (also helps long/short parsing)
                if view_url:
                    if view_url in updated_cache:
                        updated_on = _max_iso_date(updated_cache[view_url], updated_on)
                        pr = rating_cache.get(view_url) or {}
                        long_term = pr.get("long_term_rating") or long_term
                        short_term = pr.get("short_term_rating") or short_term
                        outlook = pr.get("outlook") or outlook
                    else:
                        try:
                            html = fetch_text_crisil_doc(view_url, session=session, timeout_s=30)
                            txt = html_to_text(html)
                            doc_iso = parse_date(txt) or ""
                            updated_on = _max_iso_date(rr_iso, doc_iso)
                            pr_raw = parse_ratings(txt)
                            pr = {
                                "long_term_rating": pr_raw.get("internal_rating") or "",
                                "short_term_rating": pr_raw.get("external_rating") or "",
                                "outlook": pr_raw.get("outlook") or "",
                            }
                            if not pr["long_term_rating"] or not pr["short_term_rating"]:
                                simple = parse_rating_tokens_simple(txt)
                                pr["long_term_rating"] = pr["long_term_rating"] or simple.get("long_term_rating", "")
                                pr["short_term_rating"] = pr["short_term_rating"] or simple.get("short_term_rating", "")
                                pr["outlook"] = pr["outlook"] or simple.get("outlook", "")
                            updated_cache[view_url] = updated_on
                            rating_cache[view_url] = pr
                            long_term = pr.get("long_term_rating") or long_term
                            short_term = pr.get("short_term_rating") or short_term
                            outlook = pr.get("outlook") or outlook
                        except Exception:
                            pass
                        time.sleep(max(0.0, float(sleep_s)))

                for inp in list(candidates):
                    if inp not in remaining:
                        continue
                    remaining.discard(inp)
                    out.append(
                        {
                            "input_company_name": inp,
                            "matched_company_name": name,
                            "match_type": match_type,
                            "company_code": d.get("companyCode") or "",
                            "industry_name": d.get("industryName") or "",
                            "instrument_name": "Rating Rationale",
                            "rating": rating_str,
                            "outlook": outlook,
                            "long_term_rating": long_term,
                            "short_term_rating": short_term,
                            "updated_on": updated_on,
                            "view_rating_url": view_url,
                            "date_retrieved": retrieved_at,
                        }
                    )

            if rr_nf and rr_end >= rr_nf:
                break
            rr_start = rr_end
            time.sleep(max(0.0, float(sleep_s)))

    # 2.5) Targeted filtered searches for any still-missing companies.
    # This uses CRISIL's filters to narrow the result set and then applies strict/relaxed/fuzzy matching.
    if remaining:
        for inp in list(sorted(remaining)):
            filt = {"company_name": inp}
            try:
                nf, rows = fetch_crisil_credit_ratings_list_page(session=session, start=0, end=max(100, step), filters=filt)
            except Exception:
                rows = []
            if not rows:
                continue
            by_code: Dict[str, List[Dict[str, str]]] = {}
            for r in rows:
                r["date_retrieved"] = retrieved_at
                code = r.get("company_code") or ""
                if code:
                    by_code.setdefault(code, []).append(r)

            for code, items in by_code.items():
                name = (items[0].get("company_name") or "").strip()
                if not name:
                    continue
                k_strict = _norm_company_name(name)
                k_relaxed = _norm_company_name_relaxed(name)
                inp_strict = _norm_company_name(inp)
                inp_relaxed = inp_relaxed_key.get(inp, "")

                match_type = ""
                if k_strict == inp_strict:
                    match_type = "strict_filtered"
                elif k_relaxed == inp_relaxed:
                    match_type = "relaxed_filtered"
                else:
                    # Use the same conservative fuzzy matcher, but only for this input
                    if inp in _fuzzy_match_inputs(k_relaxed):
                        match_type = "fuzzy_filtered"

                if not match_type:
                    continue

                remaining.discard(inp)
                for row in items:
                    view_url = row.get("view_rating_url") or ""
                    rating_str = row.get("rating") or ""
                    toks = parse_rating_tokens_simple(rating_str)
                    long_term = toks.get("long_term_rating", "")
                    short_term = toks.get("short_term_rating", "")
                    outlook = (row.get("outlook") or "") or toks.get("outlook", "")
                    updated_on = ""

                    if enrich_updated_on and view_url:
                        if view_url in updated_cache:
                            updated_on = updated_cache[view_url]
                            pr = rating_cache.get(view_url) or {}
                            long_term = pr.get("long_term_rating") or long_term
                            short_term = pr.get("short_term_rating") or short_term
                            outlook = pr.get("outlook") or outlook
                        else:
                            try:
                                html = fetch_text_crisil_doc(view_url, session=session, timeout_s=30)
                                txt = html_to_text(html)
                                updated_on = parse_date(txt) or ""
                                pr_raw = parse_ratings(txt)
                                pr = {
                                    "long_term_rating": pr_raw.get("internal_rating") or "",
                                    "short_term_rating": pr_raw.get("external_rating") or "",
                                    "outlook": pr_raw.get("outlook") or "",
                                }
                                if not pr["long_term_rating"] or not pr["short_term_rating"]:
                                    simple = parse_rating_tokens_simple(txt)
                                    pr["long_term_rating"] = pr["long_term_rating"] or simple.get("long_term_rating", "")
                                    pr["short_term_rating"] = pr["short_term_rating"] or simple.get("short_term_rating", "")
                                    pr["outlook"] = pr["outlook"] or simple.get("outlook", "")
                                updated_cache[view_url] = updated_on
                                rating_cache[view_url] = pr
                                long_term = pr.get("long_term_rating") or long_term
                                short_term = pr.get("short_term_rating") or short_term
                                outlook = pr.get("outlook") or outlook
                            except Exception:
                                updated_on = ""
                                updated_cache[view_url] = ""
                            time.sleep(max(0.0, float(sleep_s)))

                    out.append(
                        {
                            "input_company_name": inp,
                            "matched_company_name": name,
                            "match_type": match_type,
                            "company_code": code,
                            "industry_name": row.get("industry_name") or "",
                            "instrument_name": row.get("instrument_name") or "",
                            "rating": rating_str,
                            "outlook": outlook,
                            "long_term_rating": long_term,
                            "short_term_rating": short_term,
                            "updated_on": updated_on,
                            "view_rating_url": view_url,
                            "date_retrieved": retrieved_at,
                        }
                    )
                break
            time.sleep(max(0.0, float(sleep_s)))

    # 3) Any still missing: emit no_match stubs
    for inp in sorted(remaining):
        out.append(
            {
                "input_company_name": inp,
                "matched_company_name": "",
                "match_type": "no_match",
                "company_code": "",
                "industry_name": "",
                "instrument_name": "",
                "rating": "",
                "outlook": "",
                "long_term_rating": "",
                "short_term_rating": "",
                "updated_on": "",
                "view_rating_url": "",
                "date_retrieved": retrieved_at,
            }
        )

    return out


def _best_lt_st_from_crisil_rows(rows: Sequence[Dict[str, str]]) -> Dict[str, str]:
    """
    Aggregate CRISIL instrument rows for a single input company into best LT/ST + dates.
    Dates: CRISIL provides a single updated_on per doc; we apply it to both LT and ST when present.
    """
    # Rank order (higher is better)
    lt_order = [
        "AAA",
        "AA+",
        "AA",
        "AA-",
        "A+",
        "A",
        "A-",
        "BBB+",
        "BBB",
        "BBB-",
        "BB+",
        "BB",
        "BB-",
        "B+",
        "B",
        "B-",
        "C",
        "D",
    ]
    lt_rank = {v: i for i, v in enumerate(lt_order)}
    st_order = ["A1+", "A1", "A2+", "A2", "A3+", "A3", "A4+", "A4", "A5", "PR1+", "PR1", "PR2", "PR3", "PR4", "PR5", "F1+", "F1", "F2", "F3"]
    st_rank = {v: i for i, v in enumerate(st_order)}

    best_lt = ""
    best_st = ""
    best_lt_date = ""
    best_st_date = ""
    company_code = ""
    source_url = ""

    for r in rows:
        company_code = company_code or (r.get("company_code") or "")
        u = r.get("view_rating_url") or ""
        if u and not source_url:
            source_url = u
        upd = r.get("updated_on") or ""
        lt = (r.get("long_term_rating") or "").strip().upper()
        st = (r.get("short_term_rating") or "").strip().upper()

        if lt and (not best_lt or lt_rank.get(lt, 10_000) < lt_rank.get(best_lt, 10_000)):
            best_lt = lt
            best_lt_date = upd or best_lt_date
        if st and (not best_st or st_rank.get(st, 10_000) < st_rank.get(best_st, 10_000)):
            best_st = st
            best_st_date = upd or best_st_date

        # if we already picked rating, still prefer later dates if present
        if best_lt and upd and upd > best_lt_date:
            best_lt_date = upd
        if best_st and upd and upd > best_st_date:
            best_st_date = upd

    return {
        "company_id": company_code,
        "long_term_rating": best_lt,
        "long_term_date": best_lt_date,
        "short_term_rating": best_st,
        "short_term_date": best_st_date,
        "source_url": source_url,
    }


def final_ratings_all_agencies_for_companies(
    *,
    companies: Sequence[str],
    session: "requests.Session",
    sleep_s: float = 0.2,
    insecure: bool = False,
) -> List[Dict[str, str]]:
    """
    Produce a normalized long-format output with one row per (company, agency).
    Columns:
      company_name, agency, company_id, short_term_rating, short_term_date, long_term_rating, long_term_date
    """
    retrieved_at = dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    inputs = [c.strip() for c in companies if (c or "").strip()]
    out: List[Dict[str, str]] = []

    def emit(company: str, agency: str, company_id: str, st: str, st_date: str, lt: str, lt_date: str, source_url: str, status: str) -> None:
        out.append(
            {
                "company_name": company,
                "agency": agency,
                "company_id": company_id or "",
                "short_term_rating": st or "",
                "short_term_date": st_date or "",
                "long_term_rating": lt or "",
                "long_term_date": lt_date or "",
                "source_url": source_url or "",
                "date_retrieved": retrieved_at,
                "status": status or "",
            }
        )

    # CRISIL
    try:
        cr_rows = crisil_ratings_for_companies(
            companies=inputs,
            session=session,
            page_size_companies=500,
            sleep_s=float(sleep_s),
            enrich_updated_on=True,
        )
        by_inp: Dict[str, List[Dict[str, str]]] = {}
        for r in cr_rows:
            by_inp.setdefault(r.get("input_company_name") or "", []).append(r)
        for c in inputs:
            rows = by_inp.get(c) or []
            if not rows or (len(rows) == 1 and (rows[0].get("match_type") == "no_match")):
                emit(c, "CRISIL", "", "", "", "", "", "", "not_found")
                continue
            agg = _best_lt_st_from_crisil_rows(rows)
            emit(
                c,
                "CRISIL",
                agg.get("company_id", ""),
                agg.get("short_term_rating", ""),
                agg.get("short_term_date", ""),
                agg.get("long_term_rating", ""),
                agg.get("long_term_date", ""),
                agg.get("source_url", ""),
                "ok" if (agg.get("short_term_rating") or agg.get("long_term_rating")) else "parsed_no_signal",
            )
    except Exception:
        for c in inputs:
            emit(c, "CRISIL", "", "", "", "", "", "", "error")

    # CARE
    try:
        care_rows = care_ratings_for_companies(companies=inputs, session=session, sleep_s=float(sleep_s))
        by = {r.get("company_name", ""): r for r in care_rows if isinstance(r, dict)}
        for c in inputs:
            r = by.get(c) or {}
            upd = r.get("updated_on") or ""
            emit(
                c,
                "CARE Ratings",
                r.get("company_id") or "",
                r.get("short_term_rating") or "",
                upd,
                r.get("long_term_rating") or "",
                upd,
                r.get("source_url") or "",
                r.get("status") or ("ok" if (r.get("short_term_rating") or r.get("long_term_rating")) else "not_found"),
            )
    except Exception:
        for c in inputs:
            emit(c, "CARE Ratings", "", "", "", "", "", "", "error")

    # ICRA (RatingDetails)
    try:
        for c in inputs:
            try:
                hits = icra_search_companies(c, session=session)
                best = icra_pick_best_company(c, hits)
                if not best:
                    emit(c, "ICRA", "", "", "", "", "", "", "not_found")
                    continue
                row = fetch_icra_rating_details(session=session, company_id=str(best["id"]), company_name=str(best["label"]))
                upd = row.get("updated_on") or ""
                emit(
                    c,
                    "ICRA",
                    row.get("company_id") or "",
                    row.get("short_term_rating") or "",
                    upd,
                    row.get("long_term_rating") or "",
                    upd,
                    row.get("source_url") or "",
                    row.get("status") or "ok",
                )
            except Exception:
                emit(c, "ICRA", "", "", "", "", "", "", "error")
            time.sleep(max(0.0, float(sleep_s)))
    except Exception:
        for c in inputs:
            emit(c, "ICRA", "", "", "", "", "", "", "error")

    # India Ratings
    try:
        indr_rows = indiaratings_ratings_for_companies(companies=inputs, session=session, sleep_s=float(sleep_s))
        by = {r.get("company_name", ""): r for r in indr_rows if isinstance(r, dict)}
        for c in inputs:
            r = by.get(c) or {}
            upd = r.get("updated_on") or ""
            emit(
                c,
                "India Ratings",
                r.get("company_id") or "",
                r.get("short_term_rating") or "",
                upd,
                r.get("long_term_rating") or "",
                upd,
                r.get("source_url") or "",
                r.get("status") or ("ok" if (r.get("short_term_rating") or r.get("long_term_rating")) else "not_found"),
            )
    except Exception:
        for c in inputs:
            emit(c, "India Ratings", "", "", "", "", "", "", "error")

    # Acuité
    try:
        ac_rows = acuite_ratings_for_companies(companies=inputs, session=session, sleep_s=float(sleep_s))
        by = {r.get("company_name", ""): r for r in ac_rows if isinstance(r, dict)}
        for c in inputs:
            r = by.get(c) or {}
            upd = r.get("updated_on") or ""
            emit(
                c,
                "Acuite",
                r.get("company_id") or "",
                r.get("short_term_rating") or "",
                upd,
                r.get("long_term_rating") or "",
                upd,
                r.get("source_url") or "",
                r.get("status") or ("ok" if (r.get("short_term_rating") or r.get("long_term_rating")) else "not_found"),
            )
    except Exception:
        for c in inputs:
            emit(c, "Acuite", "", "", "", "", "", "", "error")

    return out


def final_ratings_all_agencies_for_companies_wide(
    *,
    companies: Sequence[str],
    session: "requests.Session",
    sleep_s: float = 0.2,
    insecure: bool = False,
) -> List[Dict[str, str]]:
    """
    Wide format: one row per input company, with per-agency column groups.

    Columns (per agency prefix):
      <prefix>_company_id
      <prefix>_short_term_rating
      <prefix>_short_term_date
      <prefix>_long_term_rating
      <prefix>_long_term_date

    Prefix mapping:
      CRISIL -> crisil
      CARE Ratings -> care
      ICRA -> icra
      India Ratings -> indiaratings
      Acuite -> acuite
    """
    long_rows = final_ratings_all_agencies_for_companies(
        companies=companies,
        session=session,
        sleep_s=sleep_s,
        insecure=insecure,
    )

    prefix = {
        "CRISIL": "crisil",
        "CARE Ratings": "care",
        "ICRA": "icra",
        "India Ratings": "indiaratings",
        "Acuite": "acuite",
    }
    # deterministic agency order
    agencies = ["CRISIL", "CARE Ratings", "ICRA", "India Ratings", "Acuite"]

    def _empty_row(company: str) -> Dict[str, str]:
        r: Dict[str, str] = {"company_name": company}
        for a in agencies:
            p = prefix[a]
            r[f"{p}_company_id"] = ""
            r[f"{p}_short_term_rating"] = ""
            r[f"{p}_short_term_date"] = ""
            r[f"{p}_long_term_rating"] = ""
            r[f"{p}_long_term_date"] = ""
        return r

    out_by_company: Dict[str, Dict[str, str]] = {}
    for row in long_rows:
        c = row.get("company_name") or ""
        a = row.get("agency") or ""
        if not c:
            continue
        if c not in out_by_company:
            out_by_company[c] = _empty_row(c)
        if a not in prefix:
            continue
        p = prefix[a]
        out_by_company[c][f"{p}_company_id"] = row.get("company_id") or ""
        out_by_company[c][f"{p}_short_term_rating"] = row.get("short_term_rating") or ""
        out_by_company[c][f"{p}_short_term_date"] = row.get("short_term_date") or ""
        out_by_company[c][f"{p}_long_term_rating"] = row.get("long_term_rating") or ""
        out_by_company[c][f"{p}_long_term_date"] = row.get("long_term_date") or ""

    # Emit rows in input order (stable)
    seen = set()
    wide_rows: List[Dict[str, str]] = []
    for c in [x.strip() for x in companies if (x or "").strip()]:
        if c in seen:
            continue
        seen.add(c)
        wide_rows.append(out_by_company.get(c) or _empty_row(c))
    return wide_rows


def read_companies(path: str) -> List[str]:
    companies: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name and not name.startswith("#"):
                companies.append(name)
    return companies


def _split_companies_csv(raw: Optional[str]) -> List[str]:
    """
    Split a comma/semicolon-separated company string into clean names.
    """
    if not raw:
        return []
    parts = re.split(r"[,\n;]+", raw)
    return [p.strip() for p in parts if p and p.strip()]


def collect_companies_from_args(args: argparse.Namespace) -> List[str]:
    companies: List[str] = []
    if getattr(args, "companies_file", None):
        companies.extend(read_companies(args.companies_file))
    if getattr(args, "company", None):
        companies.extend([c.strip() for c in args.company if c.strip()])
    if getattr(args, "companies_csv", None):
        for raw in args.companies_csv:
            companies.extend(_split_companies_csv(raw))
    return [c for c in companies if c]


def _emit_out_path(args: argparse.Namespace, out_path: str, default_msg: Optional[str] = None) -> None:
    """
    Print output path only when requested (useful for automation), else print default message.
    """
    if getattr(args, "print_out_path_only", False):
        print(out_path)
    elif default_msg:
        print(default_msg)


def write_csv(rows: Iterable[Dict[str, str]], out_path: str) -> None:
    fieldnames = [
        "company_name",
        "agency",
        "internal_rating",
        "external_rating",
        "updated_on",
        "outlook",
        "source_url",
        "date_retrieved",
        "status",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def iter_company_ratings(
    company: str,
    max_results_per_agency: int = 5,
    sleep_s: float = 0.8,
    insecure: bool = False,
    ca_bundle: Optional[str] = None,
) -> Iterator[Dict[str, str]]:
    """
    Streaming version of get_company_ratings(): yields rows as soon as each agency is parsed.
    Useful when you want to write to CSV incrementally (flush per row) instead of building
    one big list and writing at the end.
    """
    retrieved_at = dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    session = make_session(insecure=insecure, ca_bundle=ca_bundle)

    for agency in AGENCIES:
        query = f'"{company}" {agency.query_suffix}'
        try:
            if agency.name == "CRISIL":
                yield fetch_crisil_row(company=company, session=session, retrieved_at=retrieved_at)
                time.sleep(sleep_s)
                continue

            best: Optional[str] = None
            if agency.fixed_url:
                best = agency.fixed_url
            elif agency.use_search:
                urls = duckduckgo_search(query, max_results=max_results_per_agency, session=session)
                best = pick_best_url(urls, agency.allowed_domains)

            if not best:
                yield {
                    "company_name": company,
                    "agency": agency.name,
                    "internal_rating": "",
                    "external_rating": "",
                    "updated_on": "",
                    "outlook": "",
                    "source_url": "",
                    "date_retrieved": retrieved_at,
                    "status": "not_found",
                }
                continue

            page_html = fetch_text(best, session=session)
            text = html_to_text(page_html)
            parsed = parse_ratings(text)
            updated_on = parse_date(text) or ""

            company_mentioned = company.lower() in text.lower()

            yield {
                "company_name": company,
                "agency": agency.name,
                "internal_rating": parsed.get("internal_rating") or "",
                "external_rating": parsed.get("external_rating") or "",
                "updated_on": updated_on,
                "outlook": parsed.get("outlook") or "",
                "source_url": best,
                "date_retrieved": retrieved_at,
                "status": "ok"
                if (parsed.get("internal_rating") or parsed.get("external_rating"))
                else ("parsed_no_rating" if company_mentioned else "company_not_on_source_page"),
            }
        except SSLError:
            yield {
                "company_name": company,
                "agency": agency.name,
                "internal_rating": "",
                "external_rating": "",
                "updated_on": "",
                "outlook": "",
                "source_url": "",
                "date_retrieved": retrieved_at,
                "status": "error: SSLError (try --ca-bundle or last resort --insecure)",
            }
        except Exception as e:
            yield {
                "company_name": company,
                "agency": agency.name,
                "internal_rating": "",
                "external_rating": "",
                "updated_on": "",
                "outlook": "",
                "source_url": "",
                "date_retrieved": retrieved_at,
                "status": f"error: {type(e).__name__}",
            }
        time.sleep(sleep_s)


def write_crisil_company_matches_csv(rows: Iterable[Dict[str, str]], out_path: str) -> None:
    fieldnames = [
        "input_company_name",
        "matched_company_name",
        "match_type",
        "company_code",
        "industry_name",
        "instrument_name",
        "rating",
        "outlook",
        "long_term_rating",
        "short_term_rating",
        "updated_on",
        "view_rating_url",
        "date_retrieved",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def get_company_ratings(
    company: str,
    max_results_per_agency: int = 5,
    sleep_s: float = 0.8,
    insecure: bool = False,
    ca_bundle: Optional[str] = None,
) -> List[Dict[str, str]]:
    return list(
        iter_company_ratings(
            company,
            max_results_per_agency=max_results_per_agency,
            sleep_s=sleep_s,
            insecure=insecure,
            ca_bundle=ca_bundle,
        )
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Fetch CRISIL/CARE Edge/ICRA/India Ratings/InCred ratings for companies and export CSV."
    )
    p.add_argument(
        "--care-from-url",
        default=None,
        help="Parse a CARE Ratings press release PDF URL and print/write extracted ratings.",
    )
    p.add_argument(
        "--care-company-name",
        default="",
        help="Optional company name label for --care-from-url.",
    )
    p.add_argument(
        "--care-out-csv",
        default="",
        help="Optional output CSV path for --care-from-url (if provided, writes a 1-row CSV).",
    )
    p.add_argument(
        "--care-ratings-for-companies",
        action="store_true",
        help="Find latest CARE PR PDF per company in --companies-file (or repeated --company), parse ratings, and export CSV.",
    )
    p.add_argument(
        "--care-ratings-out-csv",
        default=f"care_company_ratings_{dt.date.today().isoformat()}.csv",
        help="Output CSV for --care-ratings-for-companies.",
    )
    p.add_argument(
        "--care-search-max-results",
        type=int,
        default=10,
        help="Max DuckDuckGo results to consider per company when locating CARE PR PDF URLs.",
    )
    p.add_argument(
        "--icra-ratings-for-companies",
        action="store_true",
        help="Fetch ICRA ratings for companies in --companies-file by resolving CompanyId via /Rating/GetRatingCompanys and scraping RatingDetails.",
    )
    p.add_argument(
        "--icra-rationale-id",
        default=None,
        help="Parse an ICRA rationale report by Id (from /Rationale/ShowRationaleReport?Id=...).",
    )
    p.add_argument(
        "--icra-company-name",
        default="",
        help="Optional company name label for --icra-rationale-id.",
    )
    p.add_argument(
        "--icra-rationale-out-csv",
        default="",
        help="Optional output CSV path for --icra-rationale-id (writes a 1-row CSV).",
    )
    p.add_argument(
        "--indiaratings-pressrelease-id",
        default=None,
        help="Parse an India Ratings press release by id (from /pressrelease/<id>).",
    )
    p.add_argument(
        "--indiaratings-company-name",
        default="",
        help="Optional company name label for --indiaratings-pressrelease-id.",
    )
    p.add_argument(
        "--indiaratings-out-csv",
        default="",
        help="Optional output CSV path for --indiaratings-pressrelease-id (writes a 1-row CSV).",
    )
    p.add_argument(
        "--indiaratings-ratings-for-companies",
        action="store_true",
        help="Find latest India Ratings press release per company in --companies-file (or repeated --company), parse ratings, and export CSV.",
    )
    p.add_argument(
        "--indiaratings-ratings-out-csv",
        default=f"indiaratings_company_ratings_{dt.date.today().isoformat()}.csv",
        help="Output CSV for --indiaratings-ratings-for-companies.",
    )
    p.add_argument(
        "--indiaratings-search-max-results",
        type=int,
        default=10,
        help="Max India Ratings issuer search hits to consider per company.",
    )
    p.add_argument(
        "--acuite-from-url",
        default=None,
        help="Parse an Acuité press release URL (connect.acuite.in/fcompany-details/...) and print/write extracted ratings.",
    )
    p.add_argument(
        "--acuite-company-name",
        default="",
        help="Optional company name label for --acuite-from-url.",
    )
    p.add_argument(
        "--acuite-out-csv",
        default="",
        help="Optional output CSV path for --acuite-from-url (writes a 1-row CSV).",
    )
    p.add_argument(
        "--acuite-ratings-for-companies",
        action="store_true",
        help="Find latest Acuité fcompany-details page per company in --companies-file (or repeated --company), parse ratings, and export CSV.",
    )
    p.add_argument(
        "--acuite-ratings-out-csv",
        default=f"acuite_company_ratings_{dt.date.today().isoformat()}.csv",
        help="Output CSV for --acuite-ratings-for-companies.",
    )
    p.add_argument(
        "--acuite-search-max-results",
        type=int,
        default=10,
        help="Max DuckDuckGo results to consider per company when locating Acuité fcompany-details URLs.",
    )
    p.add_argument(
        "--final-all-agencies",
        action="store_true",
        help="Fetch CRISIL/CARE/ICRA/IndiaRatings/Acuite ratings for companies in --companies-file (or repeated --company) and write a normalized CSV.",
    )
    p.add_argument(
        "--final-out-csv",
        default=f"final_credit_ratings_all_agencies_{dt.date.today().isoformat()}.csv",
        help="Output CSV path for --final-all-agencies.",
    )
    p.add_argument(
        "--final-wide",
        action="store_true",
        help="Write final output in wide format (one row per company; per-agency column groups).",
    )
    p.add_argument(
        "--icra-out-csv",
        default=f"icra_company_ratings_{dt.date.today().isoformat()}.csv",
        help="Output CSV for --icra-ratings-for-companies",
    )
    p.add_argument(
        "--crisil-export-ratingdocs",
        action="store_true",
        help=(
            "Enumerate all CRISIL 'View rating' document URLs published in the CRISIL public list, "
            "filtering to those under --crisil-ratingdocs-prefix, and write them to a newline-delimited file."
        ),
    )
    p.add_argument(
        "--crisil-ratingdocs-prefix",
        default="https://www.crisil.com/mnt/winshare/Ratings/RatingList/RatingDocs/",
        help="Prefix URL to filter 'View rating' document URLs (default: CRISIL RatingDocs directory).",
    )
    p.add_argument(
        "--crisil-ratingdocs-out",
        default=f"crisil_ratingdocs_urls_{dt.date.today().isoformat()}.txt",
        help="Output path for --crisil-export-ratingdocs (newline-delimited URLs).",
    )
    p.add_argument(
        "--crisil-max-companies",
        type=int,
        default=None,
        help="Optional cap on how many CRISIL list rows to scan (useful for testing).",
    )
    p.add_argument(
        "--crisil-parse-ratingdocs",
        action="store_true",
        help=(
            "Read a newline-delimited file of CRISIL RatingDocs URLs and extract short-term rating + "
            "'updated on'/'rating outstanding as on' date into a CSV."
        ),
    )
    p.add_argument(
        "--crisil-ratingdocs-urls-file",
        default="crisil_ratingdocs_urls.txt",
        help="Input newline-delimited file of RatingDocs URLs for --crisil-parse-ratingdocs.",
    )
    p.add_argument(
        "--crisil-ratingdocs-parsed-out-csv",
        default=f"crisil_ratingdocs_parsed_{dt.date.today().isoformat()}.csv",
        help="Output CSV path for --crisil-parse-ratingdocs.",
    )
    p.add_argument(
        "--crisil-ratingdocs-workers",
        type=int,
        default=8,
        help="Number of concurrent workers for --crisil-parse-ratingdocs.",
    )
    p.add_argument(
        "--crisil-ratingdocs-max-urls",
        type=int,
        default=None,
        help="Optional cap on how many RatingDocs URLs to process (useful for testing).",
    )
    p.add_argument(
        "--crisil-ratingdocs-no-resume",
        action="store_true",
        help="Disable resume behavior for --crisil-parse-ratingdocs (by default it skips already-processed URLs).",
    )
    p.add_argument(
        "--crisil-retry-ratingdocs-http-errors",
        action="store_true",
        help="Retry rows with status 'error: HTTPError' from an existing parsed RatingDocs CSV and write a refreshed CSV.",
    )
    p.add_argument(
        "--crisil-retry-in-csv",
        default="crisil_ratingdocs_parsed_all.csv",
        help="Input parsed RatingDocs CSV for --crisil-retry-ratingdocs-http-errors.",
    )
    p.add_argument(
        "--crisil-retry-out-csv",
        default=f"crisil_ratingdocs_parsed_all_retried_{dt.date.today().isoformat()}.csv",
        help="Output CSV for --crisil-retry-ratingdocs-http-errors.",
    )
    p.add_argument(
        "--crisil-retry-attempts",
        type=int,
        default=4,
        help="Attempts per URL during --crisil-retry-ratingdocs-http-errors.",
    )
    p.add_argument(
        "--crisil-retry-workers",
        type=int,
        default=8,
        help="Concurrency for --crisil-retry-ratingdocs-http-errors.",
    )
    p.add_argument(
        "--crisil-ratings-for-companies",
        action="store_true",
        help="Fetch CRISIL ratings for companies in --companies-file (or repeated --company) by scraping CRISIL public list and locally matching names.",
    )
    p.add_argument(
        "--crisil-ratings-out-csv",
        default=f"crisil_company_ratings_{dt.date.today().isoformat()}.csv",
        help="Output CSV path for --crisil-ratings-for-companies",
    )
    p.add_argument(
        "--crisil-enrich-updated-on",
        action="store_true",
        help="For --crisil-ratings-for-companies, fetch each View Rating document and parse the 'updated on'/'rating outstanding as on' date into updated_on.",
    )
    p.add_argument(
        "--crisil-page-size",
        type=int,
        default=50,
        help="Companies per page for CRISIL list scrape (the site uses 50).",
    )
    p.add_argument(
        "--companies-file",
        help="Path to a text file with one company name per line",
        required=False,
    )
    p.add_argument(
        "--companies-csv",
        action="append",
        default=None,
        help="Comma/semicolon-separated company names (repeatable).",
    )
    p.add_argument(
        "--company",
        action="append",
        help="Company name (repeatable). Example: --company 'Tata Motors Limited'",
        required=False,
    )
    p.add_argument(
        "--out",
        help="Output CSV path (generic multi-agency mode)",
        default=f"credit_ratings_{dt.date.today().isoformat()}.csv",
    )
    p.add_argument("--max-results", type=int, default=5, help="Max search results per agency")
    p.add_argument("--sleep", type=float, default=0.8, help="Sleep between requests (seconds)")
    p.add_argument(
        "--ca-bundle",
        default=None,
        help="Path to a custom CA bundle PEM file (useful for corporate TLS inspection).",
    )
    p.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS certificate verification (last resort).",
    )
    p.add_argument(
        "--print-out-path-only",
        action="store_true",
        help="Print only the output path to stdout for automation.",
    )
    args = p.parse_args(argv)

    if args.acuite_from_url:
        session = make_session(insecure=bool(args.insecure), ca_bundle=args.ca_bundle)
        row = acuite_from_url(
            session=session,
            url=str(args.acuite_from_url),
            company_name=str(args.acuite_company_name or ""),
        )
        print(json.dumps(row, indent=2, sort_keys=True))
        if args.acuite_out_csv:
            fieldnames = [
                "agency",
                "company_name",
                "long_term_rating",
                "short_term_rating",
                "outlook",
                "updated_on",
                "source_url",
                "date_retrieved",
                "status",
            ]
            with open(str(args.acuite_out_csv), "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                w.writeheader()
                w.writerow(row)
        return 0

    if bool(args.acuite_ratings_for_companies):
        companies = collect_companies_from_args(args)
        if not companies:
            p.error(
                "Provide --companies-file, --companies-csv, or at least one --company for --acuite-ratings-for-companies"
            )

        session = make_session(insecure=bool(args.insecure), ca_bundle=args.ca_bundle)
        rows = acuite_ratings_for_companies(
            companies=companies,
            session=session,
            sleep_s=float(args.sleep),
            max_search_results=int(args.acuite_search_max_results),
        )
        fieldnames = [
            "agency",
            "company_name",
            "long_term_rating",
            "short_term_rating",
            "outlook",
            "updated_on",
            "source_url",
            "date_retrieved",
            "status",
        ]
        with open(str(args.acuite_ratings_out_csv), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
        _emit_out_path(
            args,
            str(args.acuite_ratings_out_csv),
            default_msg=f"Wrote {len(rows)} rows to {args.acuite_ratings_out_csv}",
        )
        return 0

    if bool(args.final_all_agencies):
        companies = collect_companies_from_args(args)
        if not companies:
            p.error("Provide --companies-file, --companies-csv, or at least one --company for --final-all-agencies")

        session = make_session(insecure=bool(args.insecure), ca_bundle=args.ca_bundle)
        if bool(args.final_wide):
            rows = final_ratings_all_agencies_for_companies_wide(
                companies=companies,
                session=session,
                sleep_s=float(args.sleep),
                insecure=bool(args.insecure),
            )
            fieldnames = ["company_name"]
            for pfx in ["crisil", "care", "icra", "indiaratings", "acuite"]:
                fieldnames.extend(
                    [
                        f"{pfx}_company_id",
                        f"{pfx}_short_term_rating",
                        f"{pfx}_short_term_date",
                        f"{pfx}_long_term_rating",
                        f"{pfx}_long_term_date",
                    ]
                )
        else:
            rows = final_ratings_all_agencies_for_companies(
                companies=companies,
                session=session,
                sleep_s=float(args.sleep),
                insecure=bool(args.insecure),
            )
            fieldnames = [
                "company_name",
                "agency",
                "company_id",
                "short_term_rating",
                "short_term_date",
                "long_term_rating",
                "long_term_date",
                "source_url",
                "date_retrieved",
                "status",
            ]
        with open(str(args.final_out_csv), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
        _emit_out_path(
            args,
            str(args.final_out_csv),
            default_msg=f"Wrote {len(rows)} rows to {args.final_out_csv}",
        )
        return 0

    if args.indiaratings_pressrelease_id:
        session = make_session(insecure=bool(args.insecure), ca_bundle=args.ca_bundle)
        row = indiaratings_pressrelease_from_id(
            session=session,
            pressrelease_id=str(args.indiaratings_pressrelease_id),
            company_name=str(args.indiaratings_company_name or ""),
        )
        print(json.dumps(row, indent=2, sort_keys=True))
        if args.indiaratings_out_csv:
            fieldnames = [
                "agency",
                "company_name",
                "long_term_rating",
                "short_term_rating",
                "outlook",
                "updated_on",
                "source_url",
                "date_retrieved",
                "status",
            ]
            with open(str(args.indiaratings_out_csv), "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                w.writeheader()
                w.writerow(row)
        return 0

    if bool(args.indiaratings_ratings_for_companies):
        companies = collect_companies_from_args(args)
        if not companies:
            p.error(
                "Provide --companies-file, --companies-csv, or at least one --company for --indiaratings-ratings-for-companies"
            )

        session = make_session(insecure=bool(args.insecure), ca_bundle=args.ca_bundle)
        rows = indiaratings_ratings_for_companies(
            companies=companies,
            session=session,
            sleep_s=float(args.sleep),
            max_hits=int(args.indiaratings_search_max_results),
        )
        fieldnames = [
            "agency",
            "company_name",
            "long_term_rating",
            "short_term_rating",
            "outlook",
            "updated_on",
            "source_url",
            "date_retrieved",
            "status",
        ]
        with open(str(args.indiaratings_ratings_out_csv), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
        _emit_out_path(
            args,
            str(args.indiaratings_ratings_out_csv),
            default_msg=f"Wrote {len(rows)} rows to {args.indiaratings_ratings_out_csv}",
        )
        return 0

    if args.icra_rationale_id:
        session = make_session(insecure=bool(args.insecure), ca_bundle=args.ca_bundle)
        row = icra_rationale_from_id(
            session=session,
            rationale_id=str(args.icra_rationale_id),
            company_name=str(args.icra_company_name or ""),
        )
        print(json.dumps(row, indent=2, sort_keys=True))
        if args.icra_rationale_out_csv:
            fieldnames = [
                "agency",
                "company_name",
                "long_term_rating",
                "short_term_rating",
                "outlook",
                "updated_on",
                "source_url",
                "date_retrieved",
                "status",
            ]
            with open(str(args.icra_rationale_out_csv), "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                w.writeheader()
                w.writerow(row)
        return 0

    if args.care_from_url:
        session = make_session(insecure=bool(args.insecure), ca_bundle=args.ca_bundle)
        row = care_ratings_from_url(
            session=session,
            url=str(args.care_from_url),
            company_name=str(args.care_company_name or ""),
        )
        print(json.dumps(row, indent=2, sort_keys=True))
        if args.care_out_csv:
            fieldnames = [
                "agency",
                "company_name",
                "long_term_rating",
                "short_term_rating",
                "outlook",
                "updated_on",
                "source_url",
                "date_retrieved",
                "status",
            ]
            with open(str(args.care_out_csv), "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                w.writeheader()
                w.writerow(row)
        return 0

    if bool(args.care_ratings_for_companies):
        companies = collect_companies_from_args(args)
        if not companies:
            p.error("Provide --companies-file, --companies-csv, or at least one --company for --care-ratings-for-companies")

        session = make_session(insecure=bool(args.insecure), ca_bundle=args.ca_bundle)
        rows = care_ratings_for_companies(
            companies=companies,
            session=session,
            sleep_s=float(args.sleep),
            max_search_results=int(args.care_search_max_results),
        )
        fieldnames = [
            "agency",
            "company_name",
            "long_term_rating",
            "short_term_rating",
            "outlook",
            "updated_on",
            "source_url",
            "date_retrieved",
            "status",
        ]
        with open(str(args.care_ratings_out_csv), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
        _emit_out_path(
            args,
            str(args.care_ratings_out_csv),
            default_msg=f"Wrote {len(rows)} rows to {args.care_ratings_out_csv}",
        )
        return 0

    if bool(args.icra_ratings_for_companies):
        companies = collect_companies_from_args(args)
        if not companies:
            p.error("Provide --companies-file, --companies-csv, or at least one --company for --icra-ratings-for-companies")

        session = make_session(insecure=bool(args.insecure), ca_bundle=args.ca_bundle)
        fieldnames = [
            "agency",
            "company_id",
            "company_name",
            "matched_company_name",
            "long_term_rating",
            "short_term_rating",
            "updated_on",
            "source_url",
            "date_retrieved",
            "status",
        ]
        total = 0
        with open(str(args.icra_out_csv), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            f.flush()
            for c in companies:
                try:
                    hits = icra_search_companies(c, session=session)
                    best = icra_pick_best_company(c, hits)
                    if not best:
                        w.writerow(
                            {
                                "agency": "ICRA",
                                "company_id": "",
                                "company_name": c,
                                "matched_company_name": "",
                                "long_term_rating": "",
                                "short_term_rating": "",
                                "updated_on": "",
                                "source_url": "",
                                "date_retrieved": dt.datetime.now(dt.UTC)
                                .replace(microsecond=0)
                                .isoformat()
                                .replace("+00:00", "Z"),
                                "status": "not_found",
                            }
                        )
                        total += 1
                        f.flush()
                        continue

                    row = fetch_icra_rating_details(
                        session=session,
                        company_id=best["id"],
                        company_name=best["label"],
                    )
                    # keep original input in company_name, and store matched label separately
                    row["company_name"] = c
                    row["matched_company_name"] = best["label"]
                    w.writerow(row)
                    total += 1
                    f.flush()
                except Exception as e:
                    w.writerow(
                        {
                            "agency": "ICRA",
                            "company_id": "",
                            "company_name": c,
                            "matched_company_name": "",
                            "long_term_rating": "",
                            "short_term_rating": "",
                            "updated_on": "",
                            "source_url": "",
                            "date_retrieved": dt.datetime.now(dt.UTC)
                            .replace(microsecond=0)
                            .isoformat()
                            .replace("+00:00", "Z"),
                            "status": f"error: {type(e).__name__}",
                        }
                    )
                    total += 1
                    f.flush()
                time.sleep(max(0.0, float(args.sleep)))

        _emit_out_path(
            args,
            str(args.icra_out_csv),
            default_msg=f"Wrote {total} rows to {args.icra_out_csv}",
        )
        return 0

    if bool(args.crisil_retry_ratingdocs_http_errors):
        session = make_session(insecure=bool(args.insecure), ca_bundle=args.ca_bundle)
        total, http_errs, recovered = retry_http_error_rows_in_ratingdocs_csv(
            session=session,
            in_csv=str(args.crisil_retry_in_csv),
            out_csv=str(args.crisil_retry_out_csv),
            workers=int(args.crisil_retry_workers),
            attempts=int(args.crisil_retry_attempts),
        )
        _emit_out_path(
            args,
            str(args.crisil_retry_out_csv),
            default_msg=(
                f"Wrote refreshed CSV to {args.crisil_retry_out_csv} "
                f"(total={total}, http_error_rows={http_errs}, recovered={recovered})"
            ),
        )
        return 0

    if bool(args.crisil_parse_ratingdocs):
        session = make_session(insecure=bool(args.insecure), ca_bundle=args.ca_bundle)
        n = parse_crisil_ratingdocs_urls_to_csv(
            session=session,
            urls_file=str(args.crisil_ratingdocs_urls_file),
            out_csv=str(args.crisil_ratingdocs_parsed_out_csv),
            workers=int(args.crisil_ratingdocs_workers),
            max_urls=(int(args.crisil_ratingdocs_max_urls) if args.crisil_ratingdocs_max_urls is not None else None),
            resume=(not bool(args.crisil_ratingdocs_no_resume)),
        )
        _emit_out_path(
            args,
            str(args.crisil_ratingdocs_parsed_out_csv),
            default_msg=f"Wrote {n} rows to {args.crisil_ratingdocs_parsed_out_csv}",
        )
        return 0

    if bool(args.crisil_export_ratingdocs):
        session = make_session(insecure=bool(args.insecure), ca_bundle=args.ca_bundle)
        n = export_crisil_ratingdocs_urls(
            session=session,
            out_path=str(args.crisil_ratingdocs_out),
            ratingdocs_prefix_url=str(args.crisil_ratingdocs_prefix),
            page_size_companies=int(args.crisil_page_size),
            max_companies=(int(args.crisil_max_companies) if args.crisil_max_companies is not None else None),
            sleep_s=float(args.sleep),
        )
        _emit_out_path(
            args,
            str(args.crisil_ratingdocs_out),
            default_msg=f"Wrote {n} RatingDocs URLs to {args.crisil_ratingdocs_out}",
        )
        return 0

    if bool(args.crisil_ratings_for_companies):
        companies = collect_companies_from_args(args)
        if not companies:
            p.error("Provide --companies-file, --companies-csv, or at least one --company for --crisil-ratings-for-companies")
        session = make_session(insecure=bool(args.insecure), ca_bundle=args.ca_bundle)
        matched_rows = crisil_ratings_for_companies(
            companies=companies,
            session=session,
            page_size_companies=int(args.crisil_page_size),
            sleep_s=float(args.sleep),
            enrich_updated_on=bool(args.crisil_enrich_updated_on),
        )
        write_crisil_company_matches_csv(matched_rows, args.crisil_ratings_out_csv)
        _emit_out_path(
            args,
            str(args.crisil_ratings_out_csv),
            default_msg=f"Wrote {len(matched_rows)} rows to {args.crisil_ratings_out_csv}",
        )
        return 0

    # Generic multi-agency mode (requires DuckDuckGo to be reachable for non-CRISIL agencies)
    companies2 = collect_companies_from_args(args)
    if not companies2:
        p.error("Provide --companies-file, --companies-csv, or at least one --company")

    fieldnames = [
        "company_name",
        "agency",
        "internal_rating",
        "external_rating",
        "updated_on",
        "outlook",
        "source_url",
        "date_retrieved",
        "status",
    ]
    total = 0
    with open(str(args.out), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        f.flush()
        for c in companies2:
            for row in iter_company_ratings(
                c,
                max_results_per_agency=args.max_results,
                sleep_s=args.sleep,
                insecure=bool(args.insecure),
                ca_bundle=args.ca_bundle,
            ):
                w.writerow(row)
                total += 1
                # Ensure progress is persisted even if the job is interrupted mid-run.
                f.flush()
    _emit_out_path(
        args,
        str(args.out),
        default_msg=f"Wrote {total} rows to {args.out}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
