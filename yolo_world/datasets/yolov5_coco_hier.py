from mmyolo.registry import DATASETS
from mmyolo.datasets.yolov5_coco import YOLOv5CocoDataset


@DATASETS.register_module()
class HierYOLOv5CocoDataset(YOLOv5CocoDataset):
    def parse_data_info(self, raw_data_info: dict) -> dict:
        data_info = super().parse_data_info(raw_data_info)

        raw_ann_info = raw_data_info.get('raw_ann_info', [])
        instances = data_info.get('instances', [])

        new_instances = []
        for inst, ann in zip(instances, raw_ann_info):
            inst = inst.copy()
            inst['species_id'] = ann['category_id']
            inst['stage_id'] = ann.get('stage_id')
            inst['crop_id'] = ann.get('crop_id')
            inst['location_id'] = ann.get('location_id')
            new_instances.append(inst)

        data_info['instances'] = new_instances
        return data_info