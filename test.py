# import torch
# from verf import HierarchicalPromptEncoder


# model = HierarchicalPromptEncoder(
#     num_species=87,
#     num_stages=2,
#     num_crops=34,
#     num_locations=6,
#     embed_dim=512
# )

# species_ids = torch.tensor([63, 24, 85], dtype=torch.long)
# stage_ids = torch.tensor([1, 1, 0], dtype=torch.long)
# crop_ids = torch.tensor([0, 33, 17], dtype=torch.long)
# location_ids = torch.tensor([2, 2, 4], dtype=torch.long)

# fused, gates, parts = model(
#     species_ids=species_ids,
#     stage_ids=stage_ids,
#     crop_ids=crop_ids,
#     location_ids=location_ids
# )

# print("Fused shape:", fused.shape)   # [3, 512]
# print("Gates shape:", gates.shape)   # [3, 4]
# print("Gates:", gates)

import json
from pathlib import Path
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader

from verf import HierarchicalPromptEncoder


def load_json(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class HierarchicalInstanceDataset(Dataset):
    def __init__(self, json_path: str):
        self.instances = load_json(Path(json_path))

        # Optional safety filter in case any ids are missing
        self.instances = [
            x for x in self.instances
            if x.get("species_id") is not None
            and x.get("stage_id") is not None
            and x.get("crop_id") is not None
            and x.get("location_id") is not None
        ]

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        item = self.instances[idx]

        return {
            "species_id": torch.tensor(item["species_id"], dtype=torch.long),
            "stage_id": torch.tensor(item["stage_id"], dtype=torch.long),
            "crop_id": torch.tensor(item["crop_id"], dtype=torch.long),
            "location_id": torch.tensor(item["location_id"], dtype=torch.long),

            # debugging info
            "species_name": item.get("species_name"),
            "stage_name": item.get("stage_name"),
            "crop_name": item.get("crop_name"),
            "location_name": item.get("location_name"),
            "hier_prompt": item.get("hier_prompt", ""),
            "file_name": item.get("file_name", ""),
            "annotation_id": item.get("annotation_id"),
        }


def collate_fn(batch):
    return {
        "species_id": torch.stack([x["species_id"] for x in batch]),
        "stage_id": torch.stack([x["stage_id"] for x in batch]),
        "crop_id": torch.stack([x["crop_id"] for x in batch]),
        "location_id": torch.stack([x["location_id"] for x in batch]),

        "species_name": [x["species_name"] for x in batch],
        "stage_name": [x["stage_name"] for x in batch],
        "crop_name": [x["crop_name"] for x in batch],
        "location_name": [x["location_name"] for x in batch],
        "hier_prompt": [x["hier_prompt"] for x in batch],
        "file_name": [x["file_name"] for x in batch],
        "annotation_id": [x["annotation_id"] for x in batch],
    }


if __name__ == "__main__":
    train_json = r"D:\Testing VSCode\pest_small-DS\data\pest\annotations\train_instances_hierarchical.json"

    dataset = HierarchicalInstanceDataset(train_json)
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    model = HierarchicalPromptEncoder(
        num_species=87,
        num_stages=2,
        num_crops=34,
        num_locations=6,
        embed_dim=512
    )

    batch = next(iter(dataloader))

    fused, gates, parts = model(
        species_ids=batch["species_id"],
        stage_ids=batch["stage_id"],
        crop_ids=batch["crop_id"],
        location_ids=batch["location_id"]
    )

    print("Batch species ids:", batch["species_id"])
    print("Batch stage ids:", batch["stage_id"])
    print("Batch crop ids:", batch["crop_id"])
    print("Batch location ids:", batch["location_id"])
    print()
    print("Batch species names:", batch["species_name"])
    print("Batch prompts:", batch["hier_prompt"])
    print()
    print("Fused shape:", fused.shape)
    print("Gates shape:", gates.shape)
    print("Gates:", gates)