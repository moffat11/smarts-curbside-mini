import re, json, argparse

DAY_PAT = r"(MON|TUE|WED|THU|FRI|SAT|SUN)"
HOUR_PAT = r"(\d{1,2})(?::?(\d{2}))?\s*(AM|PM)?"

def to_24(h, m=None, ampm=None):
    h = int(h); m = int(m) if m else 0
    if ampm: ampm = ampm.upper()
    if ampm == "PM" and h != 12: h += 12
    if ampm == "AM" and h == 12: h = 0
    return f"{h:02d}:{m:02d}"

def parse_rules(text):
    t = text.upper().replace("\n"," ")
    # duration like "2H" / "2 HOUR"
    dur_m = re.search(r"\b(\d+)\s*(H|HOUR)\b", t)
    duration = f"{dur_m.group(1)}h" if dur_m else None

    # days: handle "MON-FRI" ranges or list of days
    days = []
    rng = re.search(fr"{DAY_PAT}\s*-\s*{DAY_PAT}", t)
    if rng:
        order = ["MON","TUE","WED","THU","FRI","SAT","SUN"]
        a,b = rng.group(1), rng.group(2)
        ai,bi = order.index(a), order.index(b)
        days = order[ai:bi+1] if ai<=bi else order[ai:]+order[:bi+1]
    else:
        days = re.findall(DAY_PAT, t) or ["MON","FRI"]

    # times like "9AM-6PM" or "09:00-18:00"
    tm = re.search(fr"{HOUR_PAT}\s*[-–]\s*{HOUR_PAT}", t)
    if tm:
        start = to_24(tm.group(1), tm.group(2), tm.group(3))
        end   = to_24(tm.group(4), tm.group(5), tm.group(6))
    else:
        start, end = "09:00", "18:00"

    direction = "→" if any(x in t for x in ["→", "➡", ">"]) else None

    return {
        "windows": [{"days": days, "start": start, "end": end}],
        "duration_limit": duration,
        "permit": None,
        "direction": direction
    }

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_txt", default="ocr/outputs/ocr_raw.txt")
    ap.add_argument("--out_json", default="ocr/outputs/rules.json")
    args = ap.parse_args()
    txt = open(args.in_txt, "r", encoding="utf-8").read()
    rules = parse_rules(txt)
    open(args.out_json, "w", encoding="utf-8").write(json.dumps(rules, indent=2))
    print("Rules saved to:", args.out_json)
