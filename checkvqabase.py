from mmengine.config import Config
from flagevalmm.registry import DATASETS

cfg = Config.fromfile("tasks/arpggrounding/arpgrounding_attribute_val.py")
ds = DATASETS.build(cfg.dataset)
print("len:", len(ds))
one = ds[0]
print("keys:", one.keys())
print("question_id:", one["question_id"])
print("type:", one["type"])
print("img_path:", one.get("img_path"))
print("question_preview:", one["question"][:120])