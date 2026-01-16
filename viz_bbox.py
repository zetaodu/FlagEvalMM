import json
import random
import re
from pathlib import Path
from PIL import Image, ImageDraw

# 改这里：attribute / relation / priority
root = Path.home() / ".cache/flagevalmm/datasets/ARPGrounding/attribute"
data = json.loads((root / "data.json").read_text())

out_dir = Path("./_viz_bbox")
out_dir.mkdir(parents=True, exist_ok=True)

def extract_phrase(question: str) -> str:
    m = re.search(r"Description:\s*(.*)$", question, flags=re.IGNORECASE)
    return (m.group(1) if m else "").strip()

def sanitize_filename(s: str, max_len: int = 80) -> str:
    s = s.strip()
    # replace path separators and other problematic chars
    s = re.sub(r"[\\/:\*\?\"<>\|\n\r\t]", "_", s)
    s = re.sub(r"\s+", " ", s)
    return s[:max_len].strip() or "NO_PHRASE"

# 抽 N 条看看
N = 20
samples = random.sample(data, k=min(N, len(data)))

for idx, ann in enumerate(samples):
    img_path = root / ann["img_path"]
    img = Image.open(img_path).convert("RGB")
    W, H = img.size

    x1, y1, x2, y2 = ann["answer"]  # normalized xyxy
    px1, py1 = int(x1 * W), int(y1 * H)
    px2, py2 = int(x2 * W), int(y2 * H)

    draw = ImageDraw.Draw(img)
    draw.rectangle([px1, py1, px2, py2], outline="red", width=3)

    phrase = sanitize_filename(extract_phrase(ann.get("question", "")))
    save_path = out_dir / f"{idx}_{ann['question_id']}_{phrase}.jpg"
    img.save(save_path)

print(f"Saved {len(samples)} visualizations to: {out_dir.resolve()}")