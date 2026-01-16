from mmengine.config import Config
from flagevalmm.registry import DATASETS

def inspect(task):
    cfg = Config.fromfile(task)
    ds = DATASETS.build(cfg.dataset)
    x = ds[0]
    print("\n==", task, "==")
    print("name:", ds.name, "len:", len(ds))
    print("keys:", x.keys())
    print("type:", x["type"])
    print("img_path_example:", x.get("img_path"))
    print("question_preview:", x["question"][:100])

inspect("tasks/refcoco/refcoco_val.py")
inspect("tasks/arpggrounding/arpgrounding_attribute_val.py")
