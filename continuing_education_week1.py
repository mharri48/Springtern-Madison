import pdfplumber
import pandas as pd
import os
import re

gradReportFolder = None
contEdTables = []
print("script started")

# order for csv
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

# normalize differences in names over years
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

    n = n.replace("the a. james clark school of engineering", "the a. james clark school of engineering")
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

    # all names standardized
    return unit_mapping.get(n, None)

# all schools in report accounting for line splits, etc.
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

#indicates a continuing education table
pattern_type_degree = re.compile(
    r"Type\s+of\s+Degree(?:\s*[/–-]?\s*Program)?\b",
    re.IGNORECASE
)
pattern_masters = re.compile(r"Masters/MBA", re.IGNORECASE)
pattern_phd = re.compile(r"Ph\.?\s*D\.?\s+or\s+Doctoral", re.IGNORECASE)
#indicates end of table
pattern_total_row = re.compile(r"(?:Grand\s+)?Total|TOTAL", re.IGNORECASE)
pattern_pct = re.compile(r"%")
pattern_total  = re.compile(r"(?:Grand\s+)?Total|TOTAL", re.IGNORECASE)

# finds all pages with continuing education tables
def is_cont_ed_table_page(text: str) -> bool:
    if not text:
        return False

    # table header
    if pattern_type_degree.search(text):
        return True

    # secondary approaches indicates cont-ed table even when Type of Degree doesnt extract
    if pattern_masters.search(text) and pattern_total_row.search(text) and pattern_pct.search(text):
        return True

    if pattern_masters.search(text) and pattern_total_row.search(text):
        return True

    return False

# turns words into lines
def build_lines_from_words(page, y_tol=2.0):
    """
    Turn extract_words() into stable line strings by grouping words with similar 'top' y.
    """
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

def extract_table_lines_word_based(page):
    # builds lines of table based on last function
    lines = build_lines_from_words(page)

    start_idx = None
    for i, ln in enumerate(lines):
        if pattern_type_degree.search(ln) or re.search(r"^TYPE\s+OF\s+DEGREE", ln, re.IGNORECASE):
            start_idx = i
            break
    if start_idx is None:
        return None

    end_idx = None
    for i in range(start_idx, len(lines)):
        if pattern_total.search(lines[i]):
            end_idx = i
            break
    if end_idx is None:
        end_idx = len(lines) - 1

    return lines[start_idx:end_idx + 1]

# extracting continuing education tables from each file
for file in os.listdir(gradReportFolder):
    if not file.endswith(".pdf"):
        continue

    year = file.split(" ")[0]
    fullPath = os.path.join(gradReportFolder, file)

    with pdfplumber.open(fullPath) as pdf:
        page_texts = [p.extract_text() or "" for p in pdf.pages]
        # memory vars
        last_school_norm = None 
        last_table_page = None
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page_texts[page_num - 1]
            if not is_cont_ed_table_page(text):
                continue

            table_lines = extract_table_lines_word_based(page)
            if not table_lines:
                continue

            ### finding school header by scanning back a few pages
            # school is real header
            school = None
            # fall back is best match if no header found
            fallback_school = None

            # scan farther back only at start of file (when we don't have a last_school_norm yet)
            lookback = 120 if last_school_norm is None else 25

            for p in range(page_num - 1, max(page_num - lookback, -1), -1):
                raw = page_texts[p] or ""
                top = "\n".join(raw.splitlines()[:40])          # only top of page
                flat_top = " ".join(top.split())

                # dont take school name from table of contents page
                if re.search(r"Table\s+of\s+Contents|Contents\b", flat_top, re.IGNORECASE):
                    continue

                m = school_pattern.search(flat_top)
                if not m:
                    continue
                
                # signals that the header is likely on the page with Survey Response Rate
                # stop scanning once found
                if re.search(r"Survey\s+Response\s+Rate", flat_top, re.IGNORECASE):
                    school = m.group(0)
                    break
                
                # if header not found but a school mentioned near table just use that
                if fallback_school is None:
                    fallback_school = m.group(0)

            if school is None:
                school = fallback_school





            # normalize school name if possible
            school_norm = normalize_unit(school)

            if not school_norm:
                # if table is split between 2 pages
                if last_school_norm and last_table_page == page_num - 1:
                    school_norm = last_school_norm
                # overall/univ wide always first
                elif last_school_norm is None:
                    school_norm = "University-Wide"
                else:
                    continue



            last_school_norm = school_norm

            contEdTables.append({
                "year": year,
                "page": page_num,
                "school": school_norm,
                "lines": table_lines
            })
            last_school_norm = school_norm
            last_table_page = page_num

            print(year, school_norm)

print(f"Found {len(contEdTables)} cont ed tables")


# Regex for CSV creation
COUNT = r"([\d,]+)"
PCT   = r"(<?\d+(?:\.\d+)?%)"
pathways = {
    "Masters": re.compile(rf"Masters/MBA\s+{COUNT}\s+{PCT}", re.IGNORECASE),
    "PhD/Doctoral": re.compile(rf"Ph\.?D\.?\s+or\s+Doctoral\s+{COUNT}\s+{PCT}", re.IGNORECASE),
    "Law": re.compile(rf"Law\b.*?\bJ\.?\s*D\.?\b.*?\s+{COUNT}\s+{PCT}", re.IGNORECASE),
    "Health Prof": re.compile(rf"H\s*ealth\s+Professional\s*\(.*?\)\s+{COUNT}\s+{PCT}", re.IGNORECASE),
    "Certificate": re.compile(rf"\bCertificate\b\s+{COUNT}\s+{PCT}", re.IGNORECASE),
    "Second Bachelor’s": re.compile(rf"Second\s+Bachelor(?:'s|’s)?(?:\s+Degree)?\s+{COUNT}\s+{PCT}", re.IGNORECASE),
    "Associate’s": re.compile(rf"Associate(?:'s|’s|s)?(?:\s+Degree)?\s+{COUNT}\s+{PCT}", re.IGNORECASE),
    "Non-degree": re.compile(rf"Non[-–]Degree\s+Seeking(?:\s*\(.*?\))?\s+{COUNT}\s+{PCT}", re.IGNORECASE),
    "Unspecified": re.compile(rf"Unspecified\s+{COUNT}\s+{PCT}", re.IGNORECASE),
    "Other": re.compile(rf"\bOther\b\s+{COUNT}\s+{PCT}", re.IGNORECASE),
    "Total": re.compile(rf"(?:Grand\s+)?Total|TOTAL", re.IGNORECASE),
}

numbers_only = re.compile(rf"^{COUNT}\s+{PCT}$")
label_with_numbers = re.compile(rf".+?\s+{COUNT}\s+{PCT}$", re.IGNORECASE)

# fixes weird spacing that has happened in documents
# fixes differences in tables across years
def normalize_line(line: str) -> str:
    line = " ".join(line.split())
    line = line.replace("Certificate/Certification", "Certificate")
    line = re.sub(r"\bH\s*ealth\b", "Health", line, flags=re.IGNORECASE)
    line = re.sub(r"\bNon[-–]Degree\b", "Non-Degree", line, flags=re.IGNORECASE)
    return line.strip()

def pair_lines_into_rows(lines):
    fixed = []
    pending_label = None
    pending_nums = None

    for raw in lines:
        ln = normalize_line(raw)

        if pattern_type_degree.search(ln) or re.search(r"^TYPE\s+OF\s+DEGREE", ln, re.IGNORECASE):
            continue

        if label_with_numbers.match(ln):
            pending_label = None
            pending_nums = None
            fixed.append(ln)
            continue

        mnums = numbers_only.match(ln)
        if mnums:
            if pending_label:
                fixed.append(f"{pending_label} {mnums.group(1)} {mnums.group(2)}")
                pending_label = None
                pending_nums = None
            else:
                pending_nums = f"{mnums.group(1)} {mnums.group(2)}"
            continue

        if pending_nums:
            fixed.append(f"{ln} {pending_nums}")
            pending_nums = None
            pending_label = None
        else:
            pending_label = ln

    return fixed

# csv creation
years = sorted({int(t["year"]) for t in contEdTables})
template_rows = [{"Year": y, "Unit": u} for y in years for u in unit_order]

data_lookup = {}

for t in contEdTables:
    year = int(t["year"])
    unit = normalize_unit(t["school"]) or t["school"]
    if not unit:
        continue

    key = (year, unit)

    fixed_rows = pair_lines_into_rows(t["lines"])

    #accumulate across pages instead of overwriting
    out = data_lookup.setdefault(key, {})

    for rline in fixed_rows:
        for k, pat in pathways.items():
            if k == "Total":
                continue
            m = pat.search(rline)
            if m:
                # only set if missing (prevents later pages wiping earlier ones)
                out.setdefault(f"{k} N", int(m.group(1).replace(",", "")))
                out.setdefault(f"{k} %", m.group(2))

        if re.search(r"(?:Grand\s+)?Total|TOTAL", rline, re.IGNORECASE):
            m = re.search(rf"{COUNT}\s+{PCT}", rline)
            if m:
                out["Total N"] = int(m.group(1).replace(",", ""))
                out["Total %"] = m.group(2)


final_rows = []
metric_names = ["Masters","PhD/Doctoral","Law","Health Prof","Certificate","Second Bachelor’s",
                "Associate’s","Non-degree","Unspecified","Other","Total"]
for base in template_rows:
    y = base["Year"]
    u = base["Unit"]
    row = dict(base)
    row.update(data_lookup.get((y, u), {}))
    final_rows.append(row)

metric_cols = []
for name in metric_names:
    metric_cols.extend([f"{name} N", f"{name} %"])

df = pd.DataFrame(final_rows)
for c in metric_cols:
    if c not in df.columns:
        df[c] = ""

df = df[["Unit","Year"] + metric_cols]
df.to_csv(None, index=False)
