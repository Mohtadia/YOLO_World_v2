import numpy as np
from mmcv.transforms import BaseTransform
from mmyolo.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadHierLabels(BaseTransform):
    def transform(self, results: dict) -> dict:
        instances = results.get('instances', None)
        if instances is None:
            return results

        species_ids = []
        stage_ids = []
        crop_ids = []
        location_ids = []

        for inst in instances:
            species_ids.append(inst.get('species_id', inst.get('bbox_label')))
            stage_ids.append(inst.get('stage_id'))
            crop_ids.append(inst.get('crop_id'))
            location_ids.append(inst.get('location_id'))

        results['gt_species_ids'] = np.array(species_ids, dtype=np.int64)
        results['gt_stage_ids'] = np.array(stage_ids, dtype=np.int64)
        results['gt_crop_ids'] = np.array(crop_ids, dtype=np.int64)
        results['gt_location_ids'] = np.array(location_ids, dtype=np.int64)
        
        # print(results['instances'][0])

        return results