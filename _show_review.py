import json
from pathlib import Path
r = json.loads(Path("data/train_review_proposed.json").read_text(encoding="utf-8"))
print("SUMMARY", json.dumps(r["summary"], indent=2))
print("\nDROPPED")
for d in r["dropped_comps"]:
    print("-", d["name"], "("+d["tribe"]+") missing", ", ".join(d["missing"]))
print("\nKEPT")
for b in r["kept_comps_with_boards"]:
    board = ", ".join(m["name"] for m in b["example_board"])
    print("-" , b["name"], "["+b["tribe"]+"/"+str(b["rank"])+"]")
    print("  board:", board)
print("\nSYNTH")
for b in r["synth_comps_with_boards"]:
    board = ", ".join(m["name"] for m in b["example_board"])
    print("-", b["name"])
    print("  cores:", ", ".join(b["core_names"]))
    print("  board:", board)
print("\nTHIN150 clean count", len(r["thin150_clean_boards"]))
for b in r["thin150_clean_boards"][:6]:
    print("thin#"+str(b["idx"]), "placement="+str(b["placement"]), "comp="+str(b["composition_id"]))
    for s in b["board"]:
        print("  ", s)
