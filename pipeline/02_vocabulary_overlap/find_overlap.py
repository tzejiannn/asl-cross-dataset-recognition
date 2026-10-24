"""
Finds the words WLASL100 and ASL Citizen share.

For each WLASL100 word (../01_wlasl_download/metadata/WLASL_asl100.json), finds the ASL Citizen signs
with the same name, with or without a version number (bed -> BED1, BED2), in ASL Citizen's list of
signs and their ASL-LEX codes (../../data/asl_citizen/ASL_Citizen/splits/*.csv). A word is kept
only if those signs have exactly one ASL-LEX code, i.e. ASL Citizen has exactly one sign for it, so
the word means the same sign in both datasets.

Outputs:
    class_list.txt    the kept words: class id \\t ASL Citizen sign \\t WLASL word
    vocabulary.json   the kept words with their ASL-LEX code, and the dropped words with the reason

Usage (from this folder):
    python find_overlap.py
"""
import csv
import json
import os
import re

WLASL100 = os.path.join("..", "01_wlasl_download", "metadata", "WLASL_asl100.json")
CITIZEN_SPLITS = os.path.join("..", "..", "data", "asl_citizen", "ASL_Citizen", "splits")

asl_lex = {}  # ASL Citizen sign -> ASL-LEX code
for s in ("train", "val", "test"):
    with open(os.path.join(CITIZEN_SPLITS, f"{s}.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            asl_lex[r["Gloss"]] = r["ASL-LEX Code"]

kept, dropped = {}, {"not in ASL Citizen": [], "several ASL Citizen signs": []}
for word in sorted(e["gloss"] for e in json.load(open(WLASL100))):
    w = re.sub(r"[^A-Z]", "", word.upper())
    signs = sorted(g for g in asl_lex if re.fullmatch(w + r"\d*", re.sub(r"[^A-Z0-9]", "", g)))
    codes = {asl_lex[g] for g in signs}
    if not codes:
        dropped["not in ASL Citizen"].append(word)
    elif len(codes) > 1:
        dropped["several ASL Citizen signs"].append(word)
    else:
        kept[word] = {"sign": signs[0], "asl_lex": codes.pop()}

with open("class_list.txt", "w", encoding="utf-8") as f:
    f.writelines(f"{k}\t{kept[w]['sign']}\t{w}\n" for k, w in enumerate(sorted(kept)))
json.dump({"kept": kept, "dropped": dropped}, open("vocabulary.json", "w"), indent=1)

print(f"kept {len(kept)} words")
for reason, words in dropped.items():
    print(f"  dropped ({reason}): {len(words)} -> {', '.join(words)}")
