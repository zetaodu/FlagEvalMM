import json
from pathlib import Path

TASK_NAME = "arpgrounding_priority_val"
DATA_ROOT = Path.home() / ".cache/flagevalmm/datasets/ARPGrounding/priority"
OUT_ROOT = Path("./_eval_dummy") / TASK_NAME
OUT_ROOT.mkdir(parents=True, exist_ok=True)

data = json.loads((DATA_ROOT / "data.json").read_text())

preds = []
for x in data:  # 先做 200 条就够测链路
    bbox = x["answer"]  # normalized xyxy (list[4])
    preds.append({
        "question_id": x["question_id"],
        # evaluator 期望 pred["answer"] 是字符串，会从里面正则抽 4 个数
        "answer": f"({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]})",
    })

(OUT_ROOT / f"{TASK_NAME}.json").write_text(json.dumps(preds, indent=2, ensure_ascii=False))
print("wrote:", OUT_ROOT / f"{TASK_NAME}.json")