from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from mmdet.structures import SampleList
from mmyolo.registry import MODELS

from .yolo_world import SimpleYOLOWorldDetector
from yolo_world.models.prompts.hier_prompt_encoder import HierarchicalPromptEncoder


@MODELS.register_module()
class HierSimpleYOLOWorldDetector(SimpleYOLOWorldDetector):
    def __init__(self,
                 *args,
                 num_species=87,
                 num_stages=2,
                 num_crops=34,
                 num_locations=6,
                 hier_embed_dim=512,
                 hier_hidden_dim=256,
                 use_hier_prompt=True,
                 hier_alpha=0.5,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.use_hier_prompt = use_hier_prompt

        self.hier_prompt_encoder = HierarchicalPromptEncoder(
            num_species=num_species,
            num_stages=num_stages,
            num_crops=num_crops,
            num_locations=num_locations,
            embed_dim=hier_embed_dim,
            hidden_dim=hier_hidden_dim
        )

        # Learnable fusion weight
        self.hier_alpha = nn.Parameter(torch.tensor(float(hier_alpha)))

    def _get_batch_attr_ids(self, batch_data_samples: SampleList):
        """
        Expect each data sample to have:
            data_sample.gt_instances.species_id
            data_sample.gt_instances.stage_id
            data_sample.gt_instances.crop_id
            data_sample.gt_instances.location_id

        For the first working version, we use one representative object per image.
        """
        species_ids = []
        stage_ids = []
        crop_ids = []
        location_ids = []

        for data_sample in batch_data_samples:
            gt_instances = data_sample.gt_instances

            # first object in image as representative
            species_ids.append(gt_instances.species_id[0])
            stage_ids.append(gt_instances.stage_id[0])
            crop_ids.append(gt_instances.crop_id[0])
            location_ids.append(gt_instances.location_id[0])

        species_ids = torch.stack(species_ids).long()
        stage_ids = torch.stack(stage_ids).long()
        crop_ids = torch.stack(crop_ids).long()
        location_ids = torch.stack(location_ids).long()

        return species_ids, stage_ids, crop_ids, location_ids

    def _fuse_prompts(self, base_txt_feats: Tensor, hier_txt_feats: Tensor) -> Tensor:
        """
        base_txt_feats: [B, N, D]
        hier_txt_feats: [B, D]

        We inject the hierarchical prompt into the species slot of each sample.
        """
        alpha = torch.sigmoid(self.hier_alpha)

        fused = base_txt_feats.clone()

        # replace the class slot that corresponds to species_id later if needed
        # for first simple version, fuse with all prompts uniformly
        hier_txt_feats = hier_txt_feats.unsqueeze(1)  # [B, 1, D]
        fused = alpha * fused + (1.0 - alpha) * hier_txt_feats
        fused = F.normalize(fused, dim=-1, p=2)

        return fused

    def extract_feat(
            self,
            batch_inputs: Tensor,
            batch_data_samples: SampleList) -> Tuple[Tuple[Tensor], Tensor]:
        """Extract image features and online hierarchical prompt features."""

        # image only
        img_feats, _ = self.backbone(batch_inputs, None)

        txt_feats = self.embeddings[None]  # [1, N, D]

        if self.adapter is not None:
            txt_feats = self.adapter(txt_feats) + txt_feats
            txt_feats = F.normalize(txt_feats, dim=-1, p=2)

        txt_feats = txt_feats.repeat(img_feats[0].shape[0], 1, 1)  # [B, N, D]

        if self.use_hier_prompt and batch_data_samples is not None:
            species_ids, stage_ids, crop_ids, location_ids = self._get_batch_attr_ids(
                batch_data_samples
            )

            hier_feats, gates, _ = self.hier_prompt_encoder(
                species_ids=species_ids,
                stage_ids=stage_ids,
                crop_ids=crop_ids,
                location_ids=location_ids
            )  # [B, D]

            txt_feats = self._fuse_prompts(txt_feats, hier_feats)

        if self.with_neck:
            if self.mm_neck:
                img_feats = self.neck(img_feats, txt_feats)
            else:
                img_feats = self.neck(img_feats)

        return img_feats, txt_feats