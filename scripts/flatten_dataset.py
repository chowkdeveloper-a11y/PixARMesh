import datasets
from datasets import DatasetDict
from collections import defaultdict


def flatten_example(example):
    keys = {
        "uid",
        "scene_id",
        "image",
        "depth",
        "K",
        "wrd2cam",
        "wrd2cam_rect",
        "rect_inv",
        "panoptic_mask",
    }
    obj_keys = {
        "bounds",
        "transforms",
        "model_ids",
        "raw_inst_ids",
        "pifu_ids",
        "inst_ids",
        "vertices",
        "faces",
    }
    result = defaultdict(list)

    for i in range(len(example["uid"])):
        objects = example["objects"][i]
        num_objects = len(objects["model_ids"])

        for j in range(num_objects):
            for k in keys:
                result[k].append(example[k][i])
            new_obj = {}
            for k in obj_keys:
                new_obj[k] = [objects[k][j]]
            result["objects"].append(new_obj)
    return result


orig_data = datasets.load_dataset("datasets/3d-front-ar-packed")
result_data = DatasetDict()
for split, data in orig_data.items():
    flattened_data = data.map(
        flatten_example,
        batched=True,
        batch_size=4,
        remove_columns=["layout"],
        num_proc=64,
    )
    result_data[split] = flattened_data
result_data.save_to_disk("datasets/3d-front-ar-packed-flattened", max_shard_size="1GB")
