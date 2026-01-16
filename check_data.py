import json
from pathlib import Path
from collections import Counter

root = Path.home() / ".cache/flagevalmm/datasets/ARPGrounding/attribute"
data = json.loads((root / "data.json").read_text())

N = min(2000, len(data))  # 想全量就改成 len(data)
need = {"question_id", "img_path", "question", "question_type", "answer", "image_width", "image_height"}

cnt = Counter()
examples = {
    "missing_keys": [],
    "missing_image": [],
    "bad_bbox_format": [],
    "bbox_out_of_range": [],
}

for i, x in enumerate(data[:N]):
    missing = need - set(x)
    if missing:
        cnt["missing_keys"] += 1
        if len(examples["missing_keys"]) < 5:
            examples["missing_keys"].append((i, list(missing)))
        continue

    img = root / x["img_path"]
    if not img.exists():
        cnt["missing_image"] += 1
        if len(examples["missing_image"]) < 5:
            examples["missing_image"].append((i, str(img)))
        continue

    bb = x["answer"]
    if (not isinstance(bb, list)) or len(bb) != 4:
        cnt["bad_bbox_format"] += 1
        if len(examples["bad_bbox_format"]) < 5:
            examples["bad_bbox_format"].append((i, bb))
        continue

    if any((v < 0 or v > 1) for v in bb):
        cnt["bbox_out_of_range"] += 1
        if len(examples["bbox_out_of_range"]) < 5:
            examples["bbox_out_of_range"].append((i, bb))
        continue

    cnt["ok"] += 1

print("checked:", N)
print("ok:", cnt["ok"])
print("missing_keys:", cnt["missing_keys"])
print("missing_image:", cnt["missing_image"])
print("bad_bbox_format:", cnt["bad_bbox_format"])
print("bbox_out_of_range:", cnt["bbox_out_of_range"])
print("total_bad:", N - cnt["ok"])

print("\nexamples:")
for k, v in examples.items():
    if v:
        print(k, "->", v)