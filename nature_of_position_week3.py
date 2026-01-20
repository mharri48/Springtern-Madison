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
PCT_ANY = re.compile(r"\(?(<?\d{1,3}(?:\.\d+)?)\s*%\)?")
COUNT_ANY = re.compile(r"\b([\d]{1,3}(?:,\d{3})+|\d+)\b")
SENT_BOUNDARY = re.compile(r"[.!?\n]")

# section header + stop
sectionHeader = re.compile(rf"\bNATURE{WS}OF{WS}POSITION\b", re.IGNORECASE)
salaryStop = re.compile(r"\bSALARY\b|\bAPPENDIX\b", re.IGNORECASE)  # stop at salary/appendix

# total response
totalN = re.compile(
    rf"("
    rf"\b(?:responders|respondents|graduates|students){WS}who{WS}completed{WS}the\b"
    rf"|"
    rf"\bBased{WS}on{WS}(?:the{WS})?\d+(?:{WS}&{WS}\d+)?{WS}survey{WS}responses?(?:{WS}respectively)?\b"
    rf"|"
    rf"\bBased{WS}on{WS}(?:the{WS})?\d+(?:{WS}&{WS}\d+)?{WS}responses?(?:{WS}respectively)?\b"
    rf"|"
    rf"\bBased{WS}on{WS}\b"
    rf")",
    re.IGNORECASE
)

# narrative (pre-2020 style)  --- DO NOT CHANGE
#directReg = re.compile(r"directly\s+aligned.*?\(\s*(\d+(?:\.\d+)?)\s*%\s*\)", re.IGNORECASE | re.DOTALL)
directReg = re.compile(r"directly\s+aligned.*?\(\s*(\d+(?:\.\d+)?)\s*%\s*\)",re.IGNORECASE | re.DOTALL)

steppingReg = re.compile(r"stepping(?:\s*-\s*|\s+)?stone.*?\(\s*(\d+(?:\.\d+)?)\s*%\s*\)",re.IGNORECASE | re.DOTALL)

paysBillsReg = re.compile(
    r"(?:pays\s+the\s+bills|not\s+at\s+all\s+related|unrelated).*?\(\s*(\d+(?:\.\d+)?)\s*%\s*\)",
    re.IGNORECASE | re.DOTALL
)

# chart-label fallback (post-2019 style)
DIRECT_LBL = re.compile(r"\bEmployment\s+is\s+directly\s+aligned\b", re.IGNORECASE)
STEP_LBL   = re.compile(r"\bEmployment\s+is\s+a\s+stepping\b|\bstepping\s*stone\b", re.IGNORECASE)
BILLS_LBL  = re.compile(
    r"\bpays\s+the\s+bills\b|\bunrelated\s+to\s+career\s+goals\b|\bPosition\s+simply\b|\bPosition\s+is\s+unrelated\b",
    re.IGNORECASE
)

percentReq = re.compile(r"\bpercent\b|%", re.IGNORECASE)

# -----------------------------
# word based line reconstruction (UNCHANGED)
# -----------------------------
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

def clip_at_stop(lines):
    out = []
    for ln in lines:
        if salaryStop.search(ln):
            break
        out.append(ln)
    return out

# -----------------------------
# PRE-2020: keep your pre-window behavior (UNCHANGED)
# -----------------------------
def extract_nature_block_with_pre(page, next_page=None, pre_lines=90, post_lines=220):
    lines = build_lines_from_words(page)
    if not lines:
        return None

    start_idx = None
    for i, ln in enumerate(lines):
        if sectionHeader.search(ln):
            start_idx = i
            break
    if start_idx is None:
        return None

    a = max(0, start_idx - pre_lines)
    b = min(len(lines), start_idx + post_lines)
    block = lines[a:b]

    if next_page is not None:
        nxt = build_lines_from_words(next_page)
        if nxt:
            block = block + nxt[:60]

    block = clip_at_stop(block)
    return block

# -----------------------------
# Sentence helpers (UNCHANGED)
# -----------------------------
def _sentence_window(text, anchor_match, max_chars=400):
    if not text or not anchor_match:
        return ""

    a0, a1 = anchor_match.start(), anchor_match.end()

    left_limit = max(0, a0 - max_chars)
    left_chunk = text[left_limit:a0]
    left_boundary_idx = None
    for m in SENT_BOUNDARY.finditer(left_chunk):
        left_boundary_idx = m.end()
    sent_start = left_limit + (left_boundary_idx if left_boundary_idx is not None else 0)

    right_limit = min(len(text), a1 + max_chars)
    right_chunk = text[a1:right_limit]
    m_right = SENT_BOUNDARY.search(right_chunk)
    sent_end = a1 + (m_right.end() if m_right else len(right_chunk))

    return text[sent_start:sent_end]

def pct_nearest_anchor_same_sentence(text, anchor_pat, stop_pat=None, max_chars=300,
                                    require_pat=None, block_pat=None, tail_chars=260, backScan=None):
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

    if m2.lastindex and m2.lastindex >= 1:
        g1 = m2.group(1)
        if g1 is not None:
            return g1

    if backScan is False:
        forward = sentence[m2.end():]
        if stop_pat:
            s = stop_pat.search(forward)
            if s:
                forward = forward[:s.start()]
        mf = PCT_ANY.search(forward)
        return mf.group(1) if mf else None

    if backScan is True:
        back = sentence[:m2.start()]
        hits = list(PCT_ANY.finditer(back[-tail_chars:]))
        return hits[-1].group(1) if hits else None

    back = sentence[:m2.start()]
    hits = list(PCT_ANY.finditer(back[-tail_chars:]))
    if hits:
        return hits[-1].group(1)

    forward = sentence[m2.end():]
    if stop_pat:
        s = stop_pat.search(forward)
        if s:
            forward = forward[:s.start()]
    mf = PCT_ANY.search(forward)
    return mf.group(1) if mf else None

def pct_nearest_anchor_billpay(text, anchor_pat, stop_pat=None, max_chars=300,
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

    back = sentence[:m2.start()]
    hits = list(PCT_ANY.finditer(back[-tail_chars:]))
    if hits:
        return hits[-1].group(1)

    forward = sentence[m2.end():]
    if stop_pat:
        s = stop_pat.search(forward)
        if s:
            forward = forward[:s.start()]
    mf = PCT_ANY.search(forward)
    return mf.group(1) if mf else None

# -----------------------------
# Percent & N helpers (UNCHANGED)
# -----------------------------
def _pct_near_line(lines, idx, back=6, forward=2):
    m = PCT_ANY.search(lines[idx])
    if m:
        return m.group(1)
    for j in range(1, back + 1):
        k = idx - j
        if k >= 0:
            m = PCT_ANY.search(lines[k])
            if m:
                return m.group(1)
    for j in range(1, forward + 1):
        k = idx + j
        if k < len(lines):
            m = PCT_ANY.search(lines[k])
            if m:
                return m.group(1)
    return None

def pct_from_chart_labels(lines):
    direct = step = bills = None
    for i, ln in enumerate(lines):
        if direct is None and DIRECT_LBL.search(ln):
            direct = _pct_near_line(lines, i)
        if step is None and STEP_LBL.search(ln):
            step = _pct_near_line(lines, i)
        if bills is None and BILLS_LBL.search(ln):
            bills = _pct_near_line(lines, i)
        if direct and step and bills:
            break
    return direct, step, bills

def count_for_keyword(lines, keyword):
    for i, ln in enumerate(lines):
        if keyword.search(ln):
            for j in range(i, max(-1, i - 3), -1):
                nums = list(COUNT_ANY.finditer(lines[j]))
                if nums:
                    return nums[0].group(1)
    return None

# =============================
# NEW: 2020-2023 split-page fix
# =============================

# build line objects with x filtering so we can isolate LEFT column
def build_line_objs_from_words(page, *, x0_min=None, x0_max=None, y_tol=2.0):
    words = page.extract_words(use_text_flow=True) or []
    rows = {}
    for w in words:
        txt = (w.get("text") or "").strip()
        if not txt:
            continue

        x0 = w.get("x0", 0.0)
        if x0_min is not None and x0 < x0_min:
            continue
        if x0_max is not None and x0 > x0_max:
            continue

        y = w["top"]
        key = round(y / y_tol) * y_tol
        rows.setdefault(key, []).append(w)

    out = []
    for y in sorted(rows.keys()):
        row_words = sorted(rows[y], key=lambda ww: ww["x0"])
        line = " ".join(ww["text"] for ww in row_words)
        line = " ".join(line.split())
        if line:
            out.append({"y": y, "text": line})
    return out

def page_looks_split(page):
    """
    Returns True if there are a lot of words on BOTH sides of the page midpoint.
    Works well for 2020-2023 split layouts vs 2024 full-width.
    """
    words = page.extract_words(use_text_flow=True) or []
    if not words:
        return False

    mid = page.width / 2.0
    left = sum(1 for w in words if w.get("x0", 0) < mid - 10)
    right = sum(1 for w in words if w.get("x0", 0) > mid + 10)

    total = left + right
    if total == 0:
        return False

    # if both sides have substantial content, it's split
    return (left / total) > 0.30 and (right / total) > 0.30

def extract_post2020_block_from_header(page, next_page=None, post_lines=240):
    """
    Non-split pages (ex: 2024): take lines starting AT 'NATURE OF POSITION'
    """
    lines = build_lines_from_words(page)
    if not lines:
        return None

    start_idx = None
    for i, ln in enumerate(lines):
        if sectionHeader.search(ln):
            start_idx = i
            break
    if start_idx is None:
        return None

    block = lines[start_idx:start_idx + post_lines]

    if next_page is not None:
        nxt = build_lines_from_words(next_page)
        if nxt:
            block = block + nxt[:60]

    block = clip_at_stop(block)
    return block


def extract_post2020_blocks_split_safe(page, next_page=None, post_lines=240):
    """
    2020-2023: Nature of Position is split into two columns.
    We want LEFT column only for the CAREER GOALS summary sentence + chart.
    We still want full-width lines for N.

    Returns (left_lines, all_lines) starting AT the header.
    """
    all_objs = build_line_objs_from_words(page)
    if not all_objs:
        return None, None

    start_idx = None
    for i, L in enumerate(all_objs):
        if sectionHeader.search(L["text"]):
            start_idx = i
            break
    if start_idx is None:
        return None, None

    a = start_idx
    b = min(len(all_objs), start_idx + post_lines)

    all_lines = [x["text"] for x in all_objs[a:b]]
    all_lines = clip_at_stop(all_lines)

    # y-range for the block
    y0 = all_objs[a]["y"] - 3
    y1 = all_objs[b - 1]["y"] + 18

    # left column cutoff
    mid = page.width / 2.0

    left_objs = build_line_objs_from_words(page, x0_max=mid)
    left_lines = [x["text"] for x in left_objs if (y0 <= x["y"] <= y1)]
    left_lines = clip_at_stop(left_lines)

    if next_page is not None:
        mid2 = next_page.width / 2.0
        left_next = build_line_objs_from_words(next_page, x0_max=mid2)
        left_lines.extend([x["text"] for x in left_next[:40]])
        left_lines = clip_at_stop(left_lines)

        all_next = build_line_objs_from_words(next_page)
        all_lines.extend([x["text"] for x in all_next[:60]])
        all_lines = clip_at_stop(all_lines)

    return left_lines, all_lines

# summary sentence parsing (career-goals only)
CAREER_GOALS = re.compile(r"\bcareer\s+goals?\b", re.IGNORECASE)
FIELD_STUDY  = re.compile(r"\bfield\s+of\s+study\b|study/major|their\s+study\b|major\b", re.IGNORECASE)

# allow "steppingstone" (no space) as well
STEPPINGSTONE_TOKEN = r"(?:stepping(?:\s*-\s*|\s+)?stone|steppingstone|stepping(?:[\s\-]*(?:\w+|\d{1,3}\s*%)){0,3}[\s\-]*stone)"

SUMMARY_CAREER_GOALS = re.compile(
    rf"career\s+goals?\s*\(\s*(\d+(?:\.\d+)?)\s*%\s*\).*?"
    rf"{STEPPINGSTONE_TOKEN}\s*\(\s*(\d+(?:\.\d+)?)\s*%\s*\)",
    re.IGNORECASE | re.DOTALL
)




def extract_direct_step_from_summary(joined_left_text):
    m = CAREER_GOALS.search(joined_left_text)
    if not m:
        return None, None

    sent = _sentence_window(joined_left_text, m, max_chars=1200)
    if not sent:
        return None, None

    # ensure it's the career-goals sentence, not the field-of-study one
    if FIELD_STUDY.search(sent):
        return None, None

    m2 = SUMMARY_CAREER_GOALS.search(sent)
    if not m2:
        return None, None

    return m2.group(1), m2.group(2)

def pays_bills_by_subtraction(direct_str, step_str):
    if direct_str is None or step_str is None:
        return None
    try:
        d = float(direct_str)
        s = float(step_str)
        pb = 100.0 - d - s
        if pb < -0.5 or pb > 100.0:
            return None
        return str(int(round(max(0.0, pb))))
    except Exception:
        return None

# -----------------------------
# Main extraction
# -----------------------------
year_unit_data = {}  # (year_str, unit) -> row dict

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

            next_page = pdf.pages[page_num] if page_num < len(pdf.pages) else None
            yr_i = int(year)

            # decide unit for this block (unchanged behavior)
            school_norm = current_unit
            if not school_norm and last_school_norm and last_table_page == page_num - 1:
                school_norm = last_school_norm
            if not school_norm and last_school_norm is None:
                school_norm = "University-Wide"
            if not school_norm:
                continue

            # -----------------------------
            # PRE-2020: DO NOT CHANGE
            # -----------------------------
            if yr_i <= 2019:
                window_lines = extract_nature_block_with_pre(page, next_page=next_page, pre_lines=90, post_lines=220)
                if not window_lines:
                    continue

                totalResponses = count_for_keyword(window_lines, totalN)
                joined = " ".join(window_lines)

                directlyAligned = pct_nearest_anchor_same_sentence(
                    joined, directReg, stop_pat=steppingReg,
                    max_chars=450, require_pat=percentReq, tail_chars=260, backScan=False
                )
                steppingStone = pct_nearest_anchor_same_sentence(
                    joined, steppingReg, stop_pat=paysBillsReg,
                    max_chars=450, require_pat=percentReq, tail_chars=260, backScan=False
                )
                paysBills = pct_nearest_anchor_billpay(
                    joined, paysBillsReg, stop_pat=salaryStop,
                    max_chars=450, require_pat=percentReq, tail_chars=260
                )

                if directlyAligned is None or steppingStone is None or paysBills is None:
                    d2, s2, b2 = pct_from_chart_labels(window_lines)
                    if directlyAligned is None:
                        directlyAligned = d2
                    if steppingStone is None:
                        steppingStone = s2
                    if paysBills is None:
                        paysBills = b2

            # -----------------------------
            # 2020+: split-page safe:
            #   - pull LEFT column only for summary sentence + chart
            #   - compute paysBills by subtraction
            # -----------------------------
            else:
                if page_looks_split(page):
                    left_lines, all_lines = extract_post2020_blocks_split_safe(
                        page, next_page=next_page, post_lines=260
                    )
                    if not left_lines or not all_lines:
                        continue

                    totalResponses = count_for_keyword(all_lines, totalN)

                    # TRY full width first (fixes 2024)
                    joined_full = " ".join(all_lines)
                    directlyAligned, steppingStone = extract_direct_step_from_summary(joined_full)

                    # fallback to left only (fixes 2020-2023)
                    if directlyAligned is None or steppingStone is None:
                        joined_left = " ".join(left_lines)
                        directlyAligned, steppingStone = extract_direct_step_from_summary(joined_left)

                else:
                    post_lines = extract_post2020_block_from_header(page, next_page=next_page, post_lines=260)
                    if not post_lines:
                        continue

                    totalResponses = count_for_keyword(post_lines, totalN)

                    joined_text = " ".join(post_lines)
                    directlyAligned, steppingStone = extract_direct_step_from_summary(joined_text)

                paysBills = pays_bills_by_subtraction(directlyAligned, steppingStone)


                # fallback: chart labels (LEFT column only), then recompute paysBills
                if directlyAligned is None or steppingStone is None or paysBills is None:
                    if page_looks_split(page):
                        d2, s2, _b2 = pct_from_chart_labels(all_lines)
                        if (d2 is None or s2 is None):
                            d2, s2, _b2 = pct_from_chart_labels(left_lines)

                    else:
                        d2, s2, _b2 = pct_from_chart_labels(post_lines)

                    if directlyAligned is None:
                        directlyAligned = d2
                    if steppingStone is None:
                        steppingStone = s2
                    if paysBills is None:
                        paysBills = pays_bills_by_subtraction(directlyAligned, steppingStone)


            # store row
            key = (year, school_norm)
            if key not in year_unit_data:
                year_unit_data[key] = {
                    "Unit": school_norm,
                    "Year": year,
                    "Directly Aligned": None,
                    "Stepping Stone": None,
                    "Pays the Bills": None,
                    "N": None
                }
            if directlyAligned == None and steppingStone == None and paysBills == None:
                totalResponses=""

            row = year_unit_data[key]
            row["Directly Aligned"] = directlyAligned
            row["Stepping Stone"] = steppingStone
            row["Pays the Bills"] = paysBills
            row["N"] = totalResponses

            last_school_norm = school_norm
            last_table_page = page_num

# csv creation
years = sorted({int(year) for (year, _unit) in year_unit_data.keys()})
template_rows = [{"Year": y, "Unit": u} for y in years for u in unit_order]

metric_cols = ["Directly Aligned", "Stepping Stone", "Pays the Bills", "N"]

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
print("Wrote:", out_path)