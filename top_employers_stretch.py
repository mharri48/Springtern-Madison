import pdfplumber
import pandas as pd
import os
import re

gradReportFolder = None

# order for csv rows
unit_order = [
    "University-Wide"
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

# section header + stop (FIXED)
sectionHeader = re.compile(rf"\bGEOGRAPHIC\s+DISTRIBUTION\b|"
                           rf"\bEMPLOYMENT\s+LOCATIONS", re.IGNORECASE)
top10 = re.compile(
    r"\bTOP\s*10\s*CITIES\b|"
    r"\bTOP\s*10\s*CITIES\s*OUTSIDE\s*OF\b|"
    r"\bAPPENDIX\b",
    re.IGNORECASE
)

# top location pattern within summary text
TOP_LOC_ITEM = re.compile(
    r"(<?\d{1,3}(?:\.\d+)?)\s*%\s*"
    r"(?:reported\s+employment\s+in\s+|in\s+)"
    r"([A-Za-z][A-Za-z.\-/&\s]{0,60}?)\s*"
    r"\(\s*([\d,]+)\s*\)",
    re.IGNORECASE
)

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
        if top10.search(ln):
            break
        out.append(ln)
    return out


# finds the closest number around a keywork
def count_for_keyword(lines, keyword):
    for i, ln in enumerate(lines):
        if keyword.search(ln):
            for j in range(i, max(-1, i - 3), -1):
                nums = list(COUNT_ANY.finditer(lines[j]))
                if nums:
                    return nums[0].group(1)
    return None

# extract lines of geograph distribution secion
def extract_geo_block_with_pre(page, next_page=None, pre_lines=40, post_lines=240):
    lines = build_lines_from_words(page)
    if next_page is not None:
        lines += build_lines_from_words(next_page)[:120]

    start = None
    for i, ln in enumerate(lines):
        if sectionHeader.search(ln):
            start = i
            break
    if start is None:
        return None

    block = lines[start:start + post_lines]
    block = clip_at_stop(block)
    return block

# returns top location and graduates N based on highest percent found
def extract_top_location_and_count(joined_text):
    best = None  # (pct_float, location, count_str)

    for m in TOP_LOC_ITEM.finditer(joined_text):
        pct_str = m.group(1)
        loc = " ".join(m.group(2).split())
        cnt = m.group(3)

        # normalize DC
        if re.fullmatch(r"d\.?\s*c\.?", loc, re.IGNORECASE):
            loc = "DC"

        pct_val = float(pct_str.replace("<", ""))

        if best is None or pct_val > best[0]:
            best = (pct_val, loc, cnt)

    if best is None:
        return None, None

    return best[1], best[2]

# -----------------------------
# main extraction
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

            # figure out school
            school_norm = current_unit
            if not school_norm and last_school_norm and last_table_page == page_num - 1:
                school_norm = last_school_norm
            if not school_norm and last_school_norm is None:
                school_norm = "University-Wide"
            if not school_norm:
                continue

            if school_norm != "University-Wide":
                continue

            geo_lines = extract_geo_block_with_pre(page, next_page=next_page, pre_lines=40, post_lines=240)
            if not geo_lines:
                continue

            joined = " ".join(geo_lines)
            top_loc, top_cnt = extract_top_location_and_count(joined)

            if top_loc is None or top_cnt is None:
                continue

            key = (year, school_norm)
            if key not in year_unit_data:
                year_unit_data[key] = {
                    "Unit": school_norm,
                    "Year": year,
                    "Location": None,
                    "Graduates N": None,
                }

            row = year_unit_data[key]
            row["Location"] = top_loc
            row["Graduates N"] = top_cnt

            last_school_norm = school_norm
            last_table_page = page_num

# csv creation
years = sorted({int(year) for (year, _unit) in year_unit_data.keys()})
template_rows = [{"Year": y, "Unit": u} for y in years for u in unit_order]

metric_cols = ["Location", "Graduates N"]

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
