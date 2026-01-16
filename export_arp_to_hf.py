import os.path as osp
import pickle
from datasets import Dataset, DatasetDict
from datasets.features import Image

ARP_ROOT = "ARPGrounding-main"
PKL_MAP = {
    "attribute": osp.join(ARP_ROOT, "arpgrounding", "attribute_vg.pkl"),
    "relation": osp.join(ARP_ROOT, "arpgrounding", "relationship_vg.pkl"),
    "priority": osp.join(ARP_ROOT, "arpgrounding", "priority_vg.pkl"),
}

def find_vg_image(vg_root: str, image_id) -> str | None:
    for sub in ("VG_100K_2", "VG_100K"):
        p = osp.join(vg_root, sub, f"{image_id}.jpg")
        if osp.exists(p):
            return p
    return None

def build_split_rows(pkl_path: str, vg_root: str):
    ann = pickle.load(open(pkl_path, "rb"))  # dict: image_id -> list[pair]
    rows = []
    for image_id, pairs in ann.items():
        img_path = find_vg_image(vg_root, image_id)
        if img_path is None:
            continue


        packed_pairs = []
        for i, pair in enumerate(pairs):
            # drop the known-bad sample: attribute_713545_0
            if image_id == 713545 and i == 0:
                continue

            pos, neg = pair[0], pair[1]
            packed_pairs.append(
                {
                    "pos_phrase": str(pos["phrase"]).lower(),
                    "neg_phrase": str(neg["phrase"]).lower(),
                    "pos_bbox_xywh": [int(pos["x"]), int(pos["y"]), int(pos["w"]), int(pos["h"])],
                    "neg_bbox_xywh": [int(neg["x"]), int(neg["y"]), int(neg["w"]), int(neg["h"])],
                }
            )

        # 如果这一张图所有 pair 都被过滤空了，就别写这张图
        if not packed_pairs:
            continue

        rows.append({"image_id": int(image_id), "image": img_path, "pairs": packed_pairs})

    return rows

def main():
    vg_root = "/media/duzt/88B8CD0A191F3702/ubuntu22/visual_genome"
    repo_id = "duzetao/ARPGrounding"   # 你的 dataset repo
    private = True                    # 建议先 private

    ds_dict = {}
    for split, pkl_path in PKL_MAP.items():
        ds = Dataset.from_list(build_split_rows(pkl_path, vg_root))
        ds = ds.cast_column("image", Image())
        ds_dict[split] = ds

    DatasetDict(ds_dict).push_to_hub(repo_id, private=private)

if __name__ == "__main__":
    main()