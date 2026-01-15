import pdfplumber
import pandas as pd
import os
import re

gradReportFolder = None

# order for csv rows
unit_order = [
    "University-Wide",
    "College of Agriculture and Natural Resources",
    "College of Arts and Humanities",
    "College of Behavioral and Social Sciences",
    "College of Computer, Mathematical, and Natural Sciences",
    "College of Education",
    "College of Information",
    "The A. James Clark School of Engineering",
    "Philip Merrill College of Journalism",
    "School of Architecture, Planning, and Preservation",
    "School of Public Health",
    "School of Public Policy",
    "The Robert H. Smith School of Business",
    "College Park Scholars",
    "Honors College",
    "Letters and Sciences",
    "Undergraduate Studies"
]
unit_mapping = {u.lower(): u for u in unit_order}

# normalize differences in unit names
def normalize_unit(name):
    if not name:
        return None
    n = " ".join(name.strip().split()).lower()
    n = n.replace("&", "and")
    n = n.replace("university wide", "university-wide")
    n = n.replace("letters & sciences", "letters and sciences")
    n = n.replace("college of information studies", "college of information")
    n = n.replace("school of architecture, planning and preservation",
                  "school of architecture, planning, and preservation")
    n = n.replace("phillip merrill college of journalism", "philip merrill college of journalism")

    n = n.replace("office of undergraduate studies", "undergraduate studies")
    n = n.replace("office of undergrad studies", "undergraduate studies")

    n = n.replace("the school of public policy", "school of public policy")
    n = n.replace("school of public policy and administration", "school of public policy")

    if "clark school of engineering" in n:
        n = "the a. james clark school of engineering"

    if n in ["overall"]:
        n = "university-wide"

    if n in ["a. james clark school of engineering", "james clark school of engineering"]:
        n = "the a. james clark school of engineering"

    if n.startswith("college of computer, mathematical"):
        n = "college of computer, mathematical, and natural sciences"
    if "school of architecture" in n:
        n = "school of architecture, planning, and preservation"

    return unit_mapping.get(n, None)

# all schools in reports
school_pattern = re.compile(
    r"("
    r"College of Agriculture and Natural Resources|"
    r"College of Arts and Humanities|"
    r"College of Behavioral and Social Sciences|"
    r"College of Computer, Mathematical, and Natural Sciences|"
    r"College of Computer, Mathematical,|"
    r"College of Computer, Mathematical|"
    r"College of Education|"
    r"College of Information Studies|"
    r"College of Information|"
    r"(?:The\s+)?A\.?\s*James\s+Clark\s+School\s+of\s+Engineering|"
    r"Clark\s+School\s+of\s+Engineering|"
    r"Philip Merrill College of Journalism|"
    r"Phillip Merrill College of Journalism|"
    r"School of Architecture, Planning, and Preservation|"
    r"SCHOOL OF ARCHITECTURE, PLANNING|"
    r"School of Architecture, Planning and Preservation|"
    r"SCHOOL OF ARCHITECTURE,\s*PLANNING\s*AND\s*PRESERVATION|"
    r"SCHOOL OF PUBLIC HEALTH|"
    r"School of Public Health|"
    r"School of Public Policy|"
    r"The School of Public Policy|"
    r"School of Public Policy and Administration|"
    r"The Robert H\. Smith School of Business|"
    r"College Park Scholars|"
    r"Honors College|"
    r"Letters\s*&\s*Sciences|"
    r"Letters and Sciences|"
    r"Overall|"
    r"University Wide|"
    r"University-Wide|"
    r"Undergraduate Studies|"
    r"Office of Undergraduate Studies|"
    r"OFFICE OF UNDERGRADUATE STUDIES|"
    r")",
    re.IGNORECASE
)

# regex helpers
WS = r"(?:[\s\u00A0]+)"
OPT_WS = r"(?:[\s\u00A0]*)"
# percent with or without parenthesis
PCT_ANY = re.compile(r"\(?(<?\d{1,3}(?:\.\d+)?)\s*%\)?")

# titles of sections
internshipHeader = re.compile(rf"\bINTERNSHIP{WS}PARTICIPATION\b", re.IGNORECASE)
appendixStop = re.compile(r"\bAPPENDIX\b", re.IGNORECASE)

# total internship percent
participation_line_pat = re.compile(rf"\bat{WS}least{WS}one{WS}internship\b", re.IGNORECASE)

## word based line reconstruction
def build_lines_from_words(page, y_tol=2.0):
    words = page.extract_words(use_text_flow=True) or []
    rows = {}
    for w in words:
        txt = (w.get("text") or "").strip()
        if not txt:
            continue
        y = w["top"]
        key = round(y / y_tol) * y_tol
        rows.setdefault(key, []).append(w)

    lines = []
    for y in sorted(rows.keys()):
        row_words = sorted(rows[y], key=lambda ww: ww["x0"])
        line = " ".join(ww["text"] for ww in row_words)
        line = " ".join(line.split())
        if line:
            lines.append(line)
    return lines

# get all lines in internship participation section
def extract_table_lines_word_based(page):
    lines = build_lines_from_words(page)

    start_idx = None
    for i, ln in enumerate(lines):
        # start from internship header
        if internshipHeader.search(ln):
            start_idx = i
            break
    if start_idx is None:
        return None

    end_idx = None
    for i in range(start_idx + 1, len(lines)):
        if appendixStop.search(lines[i]):
            end_idx = i
            break
        # if line is a unit name or appendix the stop
        if school_pattern.fullmatch(lines[i].strip()):
            end_idx = i - 1
            break

    if end_idx is None:
        end_idx = len(lines) - 1

    return lines[start_idx:end_idx + 1]

# prevents extracting data in a different section
def clip_at_new_unit_or_appendix(lines):
    out = []
    for ln in lines:
        if appendixStop.search(ln) or school_pattern.fullmatch(ln.strip()):
            break
        out.append(ln)
    return out

# find pct value on line or on a nearby line
def nearest_pct_around_line(lines, idx, max_back=2, max_forward=0):
    m = PCT_ANY.search(lines[idx])
    # return first percent
    if m:
        return m.group(1)

    # find percents above line (most common)
    for j in range(1, max_back + 1):
        if idx - j >= 0:
            m = PCT_ANY.search(lines[idx - j])
            if m:
                return m.group(1)
    # find percents below line
    for j in range(1, max_forward + 1):
        if idx + j < len(lines):
            m = PCT_ANY.search(lines[idx + j])
            if m:
                return m.group(1)

    return None

# returns nearest function around a keyword
def pct_for_keyword(lines, keyword):
    for i, ln in enumerate(lines):
        if keyword.search(ln):
            return nearest_pct_around_line(lines, i, max_back=2, max_forward=0)
    return None

# regex and helper functions for conversion section of internship participation
PCT_TOKEN_INLINE = re.compile(r"(<?\d{1,3}(?:\.\d+)?)\s*%")

# returns all percents found in a list of lines
def _all_pcts_in_lines(lines):
    vals = []
    for ln in lines:
        for m in PCT_TOKEN_INLINE.finditer(ln):
            vals.append(m.group(1))
    return vals

# returns first percent found on specified line
def _pct_near(lines, idx, back=3, forward=2):
    if idx is None:
        return None
    m = PCT_TOKEN_INLINE.search(lines[idx])
    if m:
        return m.group(1)
    for j in range(1, back + 1):
        if idx - j >= 0:
            m = PCT_TOKEN_INLINE.search(lines[idx - j])
            if m:
                return m.group(1)
    for j in range(1, forward + 1):
        if idx + j < len(lines):
            m = PCT_TOKEN_INLINE.search(lines[idx + j])
            if m:
                return m.group(1)
    return None

def _find_idx(pat, lines, start=0, end=None):
    if end is None:
        end = len(lines)
    for i in range(start, end):
        if pat.search(lines[i]):
            return i
    return None

# returns all percents in a window/range
def _pcts_window(lines, idx, win=8):
    if idx is None:
        return []
    return _all_pcts_in_lines(lines[idx:min(len(lines), idx + win)])

## helpers for choosing correct percent

# return first percent thats not exclude
def _choose_excluding(vals, exclude):
    for v in vals:
        if v not in exclude:
            return v
    return None
# return smallest value thats not exclude
def _choose_min_excluding(vals, exclude):
    nums = []
    for v in vals:
        if v in exclude:
            continue
        try:
            nums.append((float(v), v))
        except:
            continue
    if not nums:
        return None
    nums.sort(key=lambda t: t[0])
    return nums[0][1]
# reutrn biggest value in vals
def _choose_max(vals):
    nums = []
    for v in vals:
        try:
            nums.append((float(v), v))
        except:
            continue
    if not nums:
        return None
    nums.sort(key=lambda t: t[0], reverse=True)
    return nums[0][1]

# ---------- main conversion extractor ----------

# marks start of conversion section
conversionHeader = re.compile(r"\b(?:Conversion|Transition)\b.*\bFull[-–—]?Time\b", re.IGNORECASE)

#things to be extracted
noOffer = re.compile(r"\bNo\s+offer\b", re.IGNORECASE)
recievedOffer = re.compile(r"\bReceived\s+offer\b", re.IGNORECASE)
acceptedOffer = re.compile(r"\bAccepted\b.*\bFT\b", re.IGNORECASE)
offerNotAccepted = re.compile(
    r"\b(not\s+to\s+accept|not\s+accept|chose\s+not\s+to\s+accept|chose\s+not\s+accept|rejected)\b",
    re.IGNORECASE
)
PAT_PURSUED_NO = re.compile(
    r"\b(did\s+not\b.*\breceive\b.*\boffer|did\s+not\s+receive\s+an\s+offer|no\s+offer)\b",
    re.IGNORECASE
)
choseNotToPursue = re.compile(r"\bchose\b.*\bnot\b.*\bpursue\b", re.IGNORECASE)

#to be used for 2019-2024
LIST_CHOSE_NOT = re.compile(r"\bChose\s+Not\s+to\s+Pursue\b.*\bFT\b", re.IGNORECASE)
LIST_ACCEPTED  = re.compile(r"\bAccepted\b.*\bFT\b.*\bEmployment\b", re.IGNORECASE)
LIST_OFFER_REJ = re.compile(r"\bOffer\s+from\b.*\bRejected\b", re.IGNORECASE)
LIST_PURSUED_NO = re.compile(r"\bPursued\b.*\bNo\s+Offer\b", re.IGNORECASE)


# conversion data post 2019 was in pie chart format
def extract_pie_2015_2018(chart_lines):
    """
    2015–2018 pie-chart mapping (minimal, robust):
      - Avoid legend collision ("No offer") when finding pursued outcome
      - Handle lines that contain two percents like "60% 12%"
      - Allow accepted % to appear several lines after label
    """
    chart = [" ".join((ln or "").split()) for ln in chart_lines if ln and ln.strip()]

    # --- helpers ---
    def pct_list(s):
        return [m.group(1) for m in PCT_TOKEN_INLINE.finditer(s or "")]
    # Scan forward from idx to find first % meeting predicate and not excluded
    def first_pct_after(idx, forward=12, exclude=set(), predicate=None):
        if idx is None:
            return None
        end = min(len(chart), idx + forward + 1)
        for j in range(idx, end):
            vals = pct_list(chart[j])
            if not vals:
                continue
            for v in vals:
                if v in exclude:
                    continue
                if predicate and not predicate(v, chart[j]):
                    continue
                return v
        return None

    def minmax_pct_after(idx, forward=12, exclude=set()):
        """Return (min_pct, max_pct) found on the first line that contains %."""
        if idx is None:
            return (None, None)
        end = min(len(chart), idx + forward + 1)
        for j in range(idx, end):
            vals = [v for v in pct_list(chart[j]) if v not in exclude]
            if not vals:
                continue
            nums = []
            for v in vals:
                try:
                    nums.append((float(v), v))
                except:
                    pass
            if not nums:
                continue
            nums.sort(key=lambda t: t[0])
            return (nums[0][1], nums[-1][1])
        return (None, None)

    # --- 1) global legend split (can be split across lines) ---
    noOffer_LEGEND = re.compile(r"\bNo\s+offer\b", re.IGNORECASE)
    PAT_RECEIVED_OFFER_LEGEND = re.compile(r"\bReceived\s+offer\b", re.IGNORECASE)

    idx_no = _find_idx(noOffer_LEGEND, chart, 0, min(len(chart), 80))
    idx_ro = _find_idx(PAT_RECEIVED_OFFER_LEGEND, chart, 0, min(len(chart), 80))

    # No-offer legend is usually the smaller one; Received-offer usually large.
    no_offer = first_pct_after(
        idx_no, forward=6, exclude=set(),
        predicate=lambda v, ln: True
    )
    received_offer = first_pct_after(
        idx_ro, forward=8, exclude=set([no_offer]) if no_offer else set(),
        predicate=lambda v, ln: (float(v) >= 50.0)  # avoids grabbing small wrong values like 17
    )

    exclude_globals = {v for v in (no_offer, received_offer) if v}

    # --- 2) find label anchors for outcomes ---
    idx_acc = _find_idx(acceptedOffer, chart, 0, len(chart))
    idx_offer_not = _find_idx(offerNotAccepted, chart, 0, len(chart))

    # IMPORTANT: do NOT use PAT_PURSUED_NO here (it matches legend "No offer")
    PAT_PURSUED_LABEL = re.compile(r"\bPursued\b", re.IGNORECASE)
    PAT_DID_NOT_RECEIVE = re.compile(r"\bdid\s+not\b.*\breceive\b.*\boffer\b", re.IGNORECASE)

    idx_purs = _find_idx(PAT_PURSUED_LABEL, chart, 0, len(chart))
    if idx_purs is None:
        idx_purs = _find_idx(PAT_DID_NOT_RECEIVE, chart, 0, len(chart))

    idx_did = _find_idx(choseNotToPursue, chart, 0, len(chart))

    # --- 3) extract outcomes (label -> nearest % after label) ---
    # Accepted can be several lines after label; also skip legend values (17,83)
    accepted_pct = first_pct_after(
        idx_acc, forward=14, exclude=exclude_globals,
        predicate=lambda v, ln: float(v) <= 50.0  # accepted is not the big legend
    )

    # Pursued-no-offer should be small/moderate; skip legend values
    pursued_no_offer_pct = first_pct_after(
        idx_purs, forward=14, exclude=exclude_globals,
        predicate=lambda v, ln: float(v) <= 40.0
    )

    # Offer-not and Did-not often share a single line like "60% 12%"
    # If so: did_not = max, offer_not = min from that first % line after their labels.
    offer_min, offer_max = minmax_pct_after(idx_offer_not, forward=18, exclude=exclude_globals)
    did_min, did_max     = minmax_pct_after(idx_did,       forward=18, exclude=exclude_globals)

    # Prefer the line local to each label, but if either is missing, fall back to the other
    offer_not_pct = offer_min or did_min
    did_not_pursue_pct = did_max or offer_max

    out = {
        "accepted_pct": accepted_pct,
        "offer_not_pct": offer_not_pct,
        "pursued_no_offer_pct": pursued_no_offer_pct,
        "did_not_pursue_pct": did_not_pursue_pct,
    }
    return out


def extract_conversion_outcomes_from_window(window_lines, year=None):
    """
    Returns dict with:
      accepted_pct, offer_not_pct, pursued_no_offer_pct, did_not_pursue_pct
    """
    L = [" ".join((ln or "").split()) for ln in window_lines if ln and ln.strip()]

    h = _find_idx(conversionHeader, L, 0, len(L))
    if h is None:
        return {"accepted_pct": None, "offer_not_pct": None, "pursued_no_offer_pct": None, "did_not_pursue_pct": None}

    block = L[h:h+180]

    # MODE A: 2020+ list-mode
    if any(LIST_PURSUED_NO.search(x) for x in block) or any(LIST_OFFER_REJ.search(x) for x in block):
        out = {"accepted_pct": None, "offer_not_pct": None, "pursued_no_offer_pct": None, "did_not_pursue_pct": None}
        for i, ln in enumerate(block):
            if out["did_not_pursue_pct"] is None and LIST_CHOSE_NOT.search(ln):
                out["did_not_pursue_pct"] = _pct_near(block, i, back=0, forward=2)
            if out["accepted_pct"] is None and LIST_ACCEPTED.search(ln):
                out["accepted_pct"] = _pct_near(block, i, back=0, forward=2)
            if out["offer_not_pct"] is None and LIST_OFFER_REJ.search(ln):
                out["offer_not_pct"] = _pct_near(block, i, back=0, forward=2)
            if out["pursued_no_offer_pct"] is None and LIST_PURSUED_NO.search(ln):
                out["pursued_no_offer_pct"] = _pct_near(block, i, back=0, forward=2)
        

    # MODE B: pre-2020
    pre = L[max(0, h-260):h]

    did_not_pursue_pct = None
    did_idx = None
    for i in range(len(pre)-1, -1, -1):
        if choseNotToPursue.search(pre[i]):
            did_idx = i
            break
    if did_idx is not None:
        did_not_pursue_pct = _pct_near(pre, did_idx, back=5, forward=3)

    chart = block[:70]

    looks_2019_inline = any(re.search(r"\bNo\s+offer\s*,\s*\d", ln, re.IGNORECASE) for ln in chart) or \
                    any(re.search(r"\bReceived\s+offer\s*,\s*\d", ln, re.IGNORECASE) for ln in chart)


    # shared extraction windows
    idx_no_offer = _find_idx(noOffer, chart, 0, len(chart))
    no_offer_cands = _pcts_window(chart, idx_no_offer, win=6)
    no_offer_pct = no_offer_cands[0] if no_offer_cands else None

    idx_recv = _find_idx(recievedOffer, chart, 0, len(chart))
    recv_cands = _pcts_window(chart, idx_recv, win=10)
    received_offer_pct = _choose_max(recv_cands)

    idx_offer_not = _find_idx(offerNotAccepted, chart, 0, len(chart))
    offer_not_cands = _pcts_window(chart, idx_offer_not, win=10)

    idx_purs = _find_idx(PAT_PURSUED_NO, chart, 0, len(chart))
    pursued_cands = _pcts_window(chart, idx_purs, win=10)

    idx_acc = _find_idx(acceptedOffer, chart, 0, len(chart))
    accepted_cands = _pcts_window(chart, idx_acc, win=14)

    # 2019 inline handling
    if looks_2019_inline:
        idx_did_chart = _find_idx(choseNotToPursue, chart, 0, len(chart))
        did_chart_pct = _pct_near(chart, idx_did_chart, back=0, forward=3)
        if did_chart_pct:
            did_not_pursue_pct = did_chart_pct

        exclude = {v for v in [did_not_pursue_pct, no_offer_pct, received_offer_pct] if v}
        # Prefer the % nearest the "offer not accepted" label (avoid accidentally grabbing pursued=3)
        offer_not_pct = _pct_near(chart, idx_offer_not, back=4, forward=4)
        if offer_not_pct in exclude:
            offer_not_pct = None
        if offer_not_pct is None:
            offer_not_pct = _choose_min_excluding(offer_not_cands, exclude)

        pursued_no_offer_pct = _choose_min_excluding(pursued_cands, exclude)

        exclude2 = exclude | {v for v in [offer_not_pct, pursued_no_offer_pct] if v}
        accepted_pct = _choose_excluding(accepted_cands, exclude2)

        out = {
            "accepted_pct": accepted_pct,
            "offer_not_pct": offer_not_pct,
            "pursued_no_offer_pct": pursued_no_offer_pct,
            "did_not_pursue_pct": did_not_pursue_pct
        }

    # MODE C: 2015–2018 pie-chart (and any other non-inline pre-2020 chart)
    out = extract_pie_2015_2018(chart, year=year)
    if out.get("did_not_pursue_pct") is None and did_not_pursue_pct is not None:
        out["did_not_pursue_pct"] = did_not_pursue_pct
    return out


# Main extraction
# collects percent and frequency
year_unit_data = {}  # (year_str, unit) -> row dict
#marks end of a sentence
SENT_BOUNDARY = re.compile(r"[.!?\n]")

## approximates area of the sentence
def _sentence_window(text, anchor_match, max_chars=400):
    if not text or not anchor_match:
        return ""

    a0, a1 = anchor_match.start(), anchor_match.end()

    # left boundary: scan left for the last sentence end punctuation
    left_limit = max(0, a0 - max_chars)
    left_chunk = text[left_limit:a0]

    # find last boundary in left_chunk
    left_boundary_idx = None
    for m in SENT_BOUNDARY.finditer(left_chunk):
        left_boundary_idx = m.end()

    sent_start = left_limit + (left_boundary_idx if left_boundary_idx is not None else 0)

    # right boundary: scan right for the next sentence end punctuation 
    right_limit = min(len(text), a1 + max_chars)
    right_chunk = text[a1:right_limit]
    m_right = SENT_BOUNDARY.search(right_chunk)
    sent_end = a1 + (m_right.end() if m_right else len(right_chunk))

    return text[sent_start:sent_end]
# returns percent matching a starting phrase(anchor)
def pct_nearest_anchor_same_sentence(text, anchor_pat, stop_pat=None, max_chars=300,
                                    require_pat=None, block_pat=None, tail_chars=260):
    m = anchor_pat.search(text)
    if not m:
        return None

    sentence = _sentence_window(text, m, max_chars=max_chars)
    if not sentence:
        return None

    if require_pat and not require_pat.search(sentence):
        return None
    if block_pat and block_pat.search(sentence):
        return None

    m2 = anchor_pat.search(sentence)
    if not m2:
        return None

    # scanning backwards produces better results
    back = sentence[:m2.start()]
    back_tail = back[-tail_chars:]
    hits = list(PCT_ANY.finditer(back_tail))
    if hits:
        return hits[-1].group(1)

    # scan forward from anchor is match was not found scanning back
    forward = sentence[m2.end():]
    if stop_pat:
        s = stop_pat.search(forward)
        if s:
            forward = forward[:s.start()]

    mf = PCT_ANY.search(forward)
    return mf.group(1) if mf else None

# multiple charts use "paid" so must be exact
paidAnchor_strict = re.compile(r"\bpaid\s+internship(?:s)?\b", re.IGNORECASE)
# for descriptions without percents
paidNone = re.compile(
    r"\bNo\s+(?:responders|respondents|graduates|student)?\b.*?\bpaid\b.*?\binternship",
    re.IGNORECASE | re.DOTALL)
allCredit = re.compile(
    r"\bAll\s+(?:responders|respondents|graduates)\b.*?\b(?:academic\s+credit|for[-\s]*credit)\b",
    re.IGNORECASE | re.DOTALL)

# fallback if strict fails (covers “paid internship experience”, split words, etc.)
loosePaid = re.compile(r"\bpaid\b", re.IGNORECASE)
percentReq = re.compile(r"\bpercent\b|%", re.IGNORECASE)

# sentence must also contain “internship” if we’re using the loose anchor
internReq = re.compile(r"\binternship(?:s)?\b", re.IGNORECASE)

# avoid matching “unpaid” chart
unpaidBlock = re.compile(r"\bunpaid\b", re.IGNORECASE)
# percent comes before anchor
creditAnchor = re.compile(r"\bacademic\s+credit\b|\bfor[-\s]*credit\b", re.IGNORECASE)
creditStop = re.compile(r"\.", re.IGNORECASE)

freqHeader = re.compile(r"\b(?:Internship\s*Frequency|Number\s+of\s+Internships)\b",
    re.IGNORECASE)
freqStop = re.compile(r"\b(?:Paid\b|For[-\s]*Credit|Academic\s+Credit|Conversion|Transition|Full[-–—]?Time|APPENDIX)\b",
    re.IGNORECASE)

re = re.compile(r"(?is)(?:\bpercent\b|%).{0,220}\bpaid\b.{0,220}\binternship"
    r"|(?:\bpercent\b|%).{0,220}\binternship.{0,220}\bpaid\b")
PAID_SENT_STRICT = re.compile(r"(?is)\b(?:\w+-)?percent\b\s*\(\s*(<?\d{1,3}(?:\.\d+)?)\s*%\s*\)"
    r".{0,220}?\bpaid\b.{0,220}?\binternship",)
CREDIT_SENT_STRICT = re.compile(r"(?is)\b(?:\w+-)?percent\b\s*\(\s*(<?\d{1,3}(?:\.\d+)?)\s*%\s*\)"
    r".{0,260}?\b(?:academic\s+credit|for[-\s]*credit)\b",)

# leaves columns blank
INSUFFICIENT_DATA = re.compile(r"\btoo\s+few\s+responses\b.*?\bgenerate\s+statistics\b"r"|\btoo\s+few\s+responses\b"
    r"|\binsufficient\s+responses\b",re.IGNORECASE | re.DOTALL)

# find frequency table header in lines and returns that section
def get_frequency_block_from_lines(lines, max_len=45):
    start = None
    for i, ln in enumerate(lines):
        if freqHeader.search(ln):
            start = i
            break
    if start is None:
        return None

    block = []
    for ln in lines[start:min(len(lines), start + max_len)]:
        if len(block) > 0 and freqStop.search(ln):
            break
        block.append(ln)
        # contains all frequency info
    return block

# find 1,2,3+ and take first percent on that line
def pct_after_label_in_lines(lines, label_pat, forward=6):
    for i, ln in enumerate(lines):
        if label_pat.search(ln):
            m = PCT_ANY.search(ln)
            if m:
                return m.group(1)
            for j in range(1, forward + 1):
                if i + j < len(lines):
                    m = PCT_ANY.search(lines[i + j])
                    if m:
                        return m.group(1)
            return None
    return None


for file in os.listdir(gradReportFolder):
    if not file.endswith(".pdf"):
        continue

    year = file.split(" ")[0]
    fullPath = os.path.join(gradReportFolder, file)

    with pdfplumber.open(fullPath) as pdf:
        page_texts = [p.extract_text() or "" for p in pdf.pages]

        current_unit = None
        last_school_norm = None
        last_table_page = None
        # finding name of current school
        for page_num, page in enumerate(pdf.pages, start=1):
            raw = page_texts[page_num - 1] or ""
            top = "\n".join(raw.splitlines()[:40])
            flat_top = " ".join(top.split())
            if not re.search(r"Table\s+of\s+Contents|Contents\b", flat_top, re.IGNORECASE):
                m_unit = school_pattern.search(flat_top)
                if m_unit:
                    cand = normalize_unit(m_unit.group(0))
                    if cand:
                        current_unit = cand
                        last_school_norm = cand

            lines_this = extract_table_lines_word_based(page)
            if not lines_this:
                continue
            if not internshipHeader.search("\n".join(lines_this)):
                continue

            lines_next = []
            if page_num < len(pdf.pages):
                next_page = pdf.pages[page_num]  # 0-indexed
                lines_next = build_lines_from_words(next_page)
                lines_next = clip_at_new_unit_or_appendix(lines_next)

            window_lines = lines_this + lines_next
            window_text = "\n".join(window_lines)

            school_norm = current_unit
            if not school_norm and last_school_norm and last_table_page == page_num - 1:
                school_norm = last_school_norm
            if not school_norm and last_school_norm is None:
                school_norm = "University-Wide"
            if not school_norm:
                continue

            participation_pct = pct_for_keyword(window_lines, participation_line_pat)
            # --- Paid / For-credit: try line-based first, then paragraph-based fallback ---
            paid_pct=None
            credit_pct=None

            if paid_pct is None or credit_pct is None:
                joined = " ".join(window_lines)
                if INSUFFICIENT_DATA.search(joined):
                    paid_pct = None
                    credit_pct = None
                else:
                    # if "none" or "all" should be 0% or 100%
                    if paid_pct is None and paidNone.search(joined):
                        paid_pct = "0"

                    if credit_pct is None and allCredit.search(joined):
                        credit_pct = "100"

                    if paid_pct is None:
                        # using strict regex first
                        paid_pct = pct_nearest_anchor_same_sentence(
                            joined,
                            paidAnchor_strict,
                            stop_pat=creditStop,
                            max_chars=450,
                            require_pat=percentReq,
                            tail_chars=260
                        )

                    if paid_pct is None:
                        # use more relaxed if that doesnt work
                        paid_pct = pct_nearest_anchor_same_sentence(
                            joined,
                            loosePaid,
                            stop_pat=creditStop,
                            max_chars=450,
                            require_pat=re,
                            block_pat=unpaidBlock,
                            tail_chars=260
                        )

                    if credit_pct is None:
                        credit_pct = pct_nearest_anchor_same_sentence(
                            joined,
                            creditAnchor,
                            stop_pat=None,
                            max_chars=450,
                            require_pat=percentReq,
                            tail_chars=260
                        )

            # if paid and credit end up identical, re-extract using strict sentence patterns
            # prevents paid stealing credits percent
            if paid_pct and credit_pct and paid_pct == credit_pct:
                # prevents code breaking due to weird line breaks
                joined = " ".join(window_lines)

                mp = PAID_SENT_STRICT.search(joined)
                mc = CREDIT_SENT_STRICT.search(joined)

                paid_fix = mp.group(1) if mp else None
                credit_fix = mc.group(1) if mc else None

                # prefer fixes if they actually separate the values
                if paid_fix and paid_fix != credit_pct:
                    paid_pct = paid_fix
                if credit_fix and credit_fix != paid_pct:
                    credit_pct = credit_fix

                

            # extract internship frequency percents
            one_label = re.compile(rf"\b1\b{OPT_WS}(?:Internship(?:s)?)?\b", re.IGNORECASE)
            two_label = re.compile(rf"\b2\b{OPT_WS}(?:Internship(?:s)?)?\b", re.IGNORECASE)
            three_label = re.compile(rf"^\s*3{OPT_WS}\+(?=\s|$){OPT_WS}(?:Internship(?:s)?)?\b",re.IGNORECASE)

            one_pct = None
            two_pct = None
            three_pct = None

            freq_block = get_frequency_block_from_lines(window_lines)
            if freq_block:
                one_pct = pct_after_label_in_lines(freq_block, one_label, forward=6)
                two_pct = pct_after_label_in_lines(freq_block, two_label, forward=6)
                three_pct = pct_after_label_in_lines(freq_block, three_label, forward=6)
            

            # conversion outcomes
            conv = extract_conversion_outcomes_from_window(
                window_lines,
                year=year,
            )

            # SAFETY NET: never allow None to crash the script
            if conv is None:
                conv = {
                    "accepted_pct": None,
                    "offer_not_pct": None,
                    "pursued_no_offer_pct": None,
                    "did_not_pursue_pct": None
                }


            accepted_pct = conv.get("accepted_pct")
            offer_not_pct = conv.get("offer_not_pct")
            pursued_no_offer_pct = conv.get("pursued_no_offer_pct")
            did_not_pursue_pct = conv.get("did_not_pursue_pct")

            
            key = (year, school_norm)
            if key not in year_unit_data:
                year_unit_data[key] = {
                    "Unit": school_norm,
                    "Year": year,
                    "Internship Participation %": None,
                    "1 Internship %": None,
                    "2 Internships %": None,
                    "3+ Internships %": None,
                    "Paid Internship %": None,
                    "For-Credit Internship %": None,
                    "Accepted FT with Internship Employer %": None,
                    "Received FT Offer (Not Accepted) %": None,
                    "Pursued FT (No Offer) %": None,
                    "Did Not Pursue FT with Host %": None,
                }
            if(three_pct == None and (one_pct != None and two_pct !=None)):
                try:
                    o = float(one_pct)
                    t = float(two_pct)
                    val = 100.0 - o - t

                    # guard against PDF noise
                    if 0 <= val <= 100:
                        # round to nearest integer to match report style
                        three_pct = str(int(round(val)))
                except Exception:
                    pass

            row = year_unit_data[key]
            row["Internship Participation %"] = participation_pct
            row["1 Internship %"] = one_pct
            row["2 Internships %"] = two_pct
            row["3+ Internships %"] = three_pct
            row["Paid Internship %"] = paid_pct
            row["For-Credit Internship %"] = credit_pct
            row["Accepted FT with Internship Employer %"] = accepted_pct
            row["Received FT Offer (Not Accepted) %"] = offer_not_pct
            row["Pursued FT (No Offer) %"] = pursued_no_offer_pct
            row["Did Not Pursue FT with Host %"] = did_not_pursue_pct

            last_school_norm = school_norm
            last_table_page = page_num

# csv creation
years = sorted({int(year) for (year, _unit) in year_unit_data.keys()})
template_rows = [{"Year": y, "Unit": u} for y in years for u in unit_order]

metric_cols = [
    "Internship Participation %",
    "1 Internship %",
    "2 Internships %",
    "3+ Internships %",
    "Paid Internship %",
    "For-Credit Internship %",
    "Accepted FT with Internship Employer %",
    "Received FT Offer (Not Accepted) %",
    "Pursued FT (No Offer) %",
    "Did Not Pursue FT with Host %",
]

final_rows = []
for base in template_rows:
    yr = base["Year"]
    unit = base["Unit"]
    key = (str(yr), unit)

    found = year_unit_data.get(key, {})
    row = dict(base)
    for col in metric_cols:
        row[col] = found.get(col, "") or ""
    final_rows.append(row)

df = pd.DataFrame(final_rows)
for c in metric_cols:
    if c not in df.columns:
        df[c] = ""

df = df[["Unit", "Year"] + metric_cols]
out_path = None
df.to_csv(out_path, index=False)