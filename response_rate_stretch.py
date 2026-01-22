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
SENT_BOUNDARY = re.compile(r"[.!?]")

stop=re.compile(rf"\bNATURE{WS}OF{WS}POSITION\b")

sectionHeader= re.compile(rf"\bSURVEY{WS}RESPONSE{WS}RATE\b", re.IGNORECASE)
totalPlacementReg = re.compile(
    rf"\b(?:TOTAL{WS}PLACEMENT|CAREER{WS}OUTCOMES{WS}RATE)\b",
    re.IGNORECASE
)
responseRateReg = re.compile(
    rf"(?:\bSURVEY{WS}RESPONSE{WS}RATE{WS}KNOWLEDGE{WS}RATE{WS}(<?\d+(?:\.\d+)?)\s*%|\bSURVEY{WS}RESPONSE{WS}RATE\b)",
    re.IGNORECASE
)
knowledgeRateReg = re.compile(
    rf"(?:\bSURVEY{WS}RESPONSE{WS}RATE{WS}KNOWLEDGE{WS}RATE{WS}<?\d+(?:\.\d+)?\s*%{WS}(<?\d+(?:\.\d+)?)\s*%|\bKNOWLEDGE{WS}RATE\b)",
    re.IGNORECASE
)



percentReq = re.compile(r"\bpercent\b|%", re.IGNORECASE)

# word based line reconstruction 
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
        if stop.search(ln):
            break
        out.append(ln)
    return out

## extracts nature of position block from pdf
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

# estimates area of sentence
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

# extracts percent closest to the anchor phrase
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

# Percent & N helpers
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


# main extraction
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

            # figure out school
            school_norm = current_unit
            if not school_norm and last_school_norm and last_table_page == page_num - 1:
                school_norm = last_school_norm
            if not school_norm and last_school_norm is None:
                school_norm = "University-Wide"
            if not school_norm:
                continue
            
            window_lines = extract_nature_block_with_pre(page, next_page=next_page, pre_lines=90, post_lines=220)
            if not window_lines:
                continue

            joined = " ".join(window_lines)
            if yr_i >= 2020:
                print("\n--- DEBUG RATE BLOCK ---")
                print("year", year, "page", page_num, "unit", school_norm)
                print("has SURVEY RESPONSE RATE:", bool(responseRateReg.search(joined)))
                print("has KNOWLEDGE RATE:", bool(knowledgeRateReg.search(joined)))
                print("has TOTAL PLACEMENT / CAREER OUTCOMES RATE:", bool(totalPlacementReg.search(joined)))
                print("first 600 chars:\n", joined[:600])
                print("--- END DEBUG ---\n")

            responseRate = pct_nearest_anchor_same_sentence(
                    joined, responseRateReg, stop_pat=totalPlacementReg,
                    max_chars=450, require_pat=percentReq, tail_chars=260, backScan=False
                )
            if responseRate is None:
                # fallback: grab the % near the label in the raw line list (handles % on next line)
                for i, ln in enumerate(window_lines):
                    if responseRateReg.search(ln):
                        responseRate = _pct_near_line(window_lines, i, back=0, forward=6)
                        break

            knowledgeRate = pct_nearest_anchor_same_sentence(
                    joined, knowledgeRateReg, stop_pat=totalPlacementReg,
                    max_chars=450, require_pat=percentReq, tail_chars=260, backScan=False
                )
            totalPlacement=pct_nearest_anchor_same_sentence(
                    joined, totalPlacementReg, stop_pat=stop,
                    max_chars=450, require_pat=percentReq, tail_chars=260, backScan=False
                )
            
            # store row
            key = (year, school_norm)
            if key not in year_unit_data:
                year_unit_data[key] = {
                    "Unit": school_norm,
                    "Year": year,
                    "Directly Aligned": None,
                    "Stepping Stone": None,
                    "Pays the Bills": None,
                    "Directly Related": None,
                    "Utilizes Knowledge/Skills":None,
                    "Not Related": None,
                    "N": None
                }

            row = year_unit_data[key]
            row["Response Rate"] = responseRate
            row["Knowledge Rate"] = knowledgeRate
            row["Total Placement"] = totalPlacement

            last_school_norm = school_norm
            last_table_page = page_num

# csv creation
years = sorted({int(year) for (year, _unit) in year_unit_data.keys()})
template_rows = [{"Year": y, "Unit": u} for y in years for u in unit_order]

metric_cols = ["Response Rate", "Knowledge Rate", "Total Placement"]

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