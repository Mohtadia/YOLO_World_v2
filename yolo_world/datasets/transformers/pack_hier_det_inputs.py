import numpy as np
import torch
from mmcv.transforms import BaseTransform
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData
from mmyolo.registry import TRANSFORMS


@TRANSFORMS.register_module()
class PackHierDetInputs(BaseTransform):
    def __init__(self, meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                                  'scale_factor', 'flip', 'flip_direction',
                                  'pad_param', 'texts')):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        packed = {}

        img = results['img']
        if not isinstance(img, np.ndarray):
            raise TypeError(f"results['img'] must be np.ndarray, got {type(img)}")

        if img.ndim == 2:
            img = img[..., None]

        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        packed['inputs'] = img

        data_sample = DetDataSample()
        gt_instances = InstanceData()

        if 'gt_bboxes' in results:
            gt_bboxes = results['gt_bboxes']
            # keep BaseBoxes object if it already exists
            if hasattr(gt_bboxes, 'tensor'):
                gt_instances.bboxes = gt_bboxes
            else:
                gt_instances.bboxes = torch.as_tensor(gt_bboxes, dtype=torch.float32)

        if 'gt_bboxes_labels' in results:
            gt_instances.labels = torch.as_tensor(
                results['gt_bboxes_labels'], dtype=torch.long)

        if 'gt_ignore_flags' in results:
            gt_instances.ignore_flags = torch.as_tensor(
                results['gt_ignore_flags'], dtype=torch.bool)

        if 'gt_species_ids' in results:
            gt_instances.species_id = torch.as_tensor(
                results['gt_species_ids'], dtype=torch.long)

        if 'gt_stage_ids' in results:
            gt_instances.stage_id = torch.as_tensor(
                results['gt_stage_ids'], dtype=torch.long)

        if 'gt_crop_ids' in results:
            gt_instances.crop_id = torch.as_tensor(
                results['gt_crop_ids'], dtype=torch.long)

        if 'gt_location_ids' in results:
            gt_instances.location_id = torch.as_tensor(
                results['gt_location_ids'], dtype=torch.long)

        data_sample.gt_instances = gt_instances

        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)

        packed['data_samples'] = data_sample
        return packed