import gzip, json

PATH = r"C:\data\goodreads\goodreads_reviews_dedup.json.gz"
with gzip.open(PATH, "rt", encoding="utf-8") as f:
    rec = json.loads(f.readline())
print("fields:", list(rec.keys()))
print("first record:", str(rec)[:600])