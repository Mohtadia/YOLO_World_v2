_base_ = '../pretrain/yolo_world_v2_s.py'

# ---------------------------------
# 1. Your 87 pest species classes
# ---------------------------------
class_name = (
    "cutworm moth",
    "tobacco cutworm moth",
    "tropical armyworm moth",
    "saw-legged coreid bug",
    "onion armyworm moth",
    "corn earworm moth",
    "rice armyworm",
    "spotted mirid bug",
    "large black scarab beetle",
    "cotton bollworm",
    "peach gall aphid",
    "small mirid bug",
    "perilla leafroller moth",
    "spotted lanternfly",
    "brown-winged planthopper",
    "soybean aphid",
    "spotted lady beetle",
    "potato tuber moth",
    "potato aphid",
    "ground cherry coreid bug",
    "sweet potato hornworm",
    "foxtail millet aphid",
    "diamondback moth",
    "cabbage white butterfly",
    "spotted cotton leafroller moth",
    "pumpkin fruit fly",
    "american serpentine leafminer",
    "root maggot fly",
    "onion leafminer fly",
    "onion leaf-feeding beetle",
    "onion leafminer moth",
    "northern jewel bug",
    "radish sawfly",
    "cabbage shoot moth",
    "narrow-thorax leaf beetle",
    "black cutworm owlet moth",
    "black cutworm moth",
    "tobacco budworm",
    "discolored mirid bug",
    "rice leafhopper",
    "flea beetle",
    "persimmon fruit moth",
    "persimmon cottony cushion scale",
    "euscaphis tree stink bug",
    "small patterned leafroller moth",
    "far eastern slug moth",
    "fall webworm",
    "white toxic moth",
    "citrus leaf miner",
    "four-spotted mugwort moth",
    "icerya scale insect",
    "tea leafroller moth",
    "arrow scale insect",
    "citrus weevil",
    "spotted longhorn beetle",
    "grass shield bug",
    "swallowtail butterfly",
    "small pear psyllid",
    "peach shoot moth",
    "brown katydid",
    "peach clearwing moth",
    "apricot leafroller moth",
    "peach fruit moth",
    "apple leaf miner",
    "mealybug",
    "apricot leaf sawfly",
    "grape mottled leafroller moth",
    "grape leafroller moth species",
    "grape clearwing moth",
    "ring-patterned blind stink bug",
    "shanxi grasshopper",
    "peach moth",
    "greenhouse whitefly",
    "corn-bordered aphid",
    "barley aphid",
    "red-patterened stink bug",
    "brown-winged stink bug",
    "spirea aphid",
    "grape hawkmoth",
    "dew leafhopper",
    "cotton aphid",
    "pea leaf miner",
    "white-hindwing noctuid moth",
    "grape tiger longhorn beetle",
    "perilla aphid",
    "black node gall aphid",
    "green peach aphid",
)

num_classes = len(class_name)

# ---------------------------------
# 2. Training settings
# ---------------------------------
max_epochs = 100
close_mosaic_epochs = 5
save_epoch_intervals = 5

# ---------------------------------
# 3. Dataset paths
# ---------------------------------
data_root = '/mnt/d/Testing VSCode/pest_small-DS/data/pest/'

train_ann_file = 'annotations/train_with_attributes.json'
train_data_prefix = 'train/images/'

val_ann_file = 'annotations/val_with_attributes.json'
val_data_prefix = 'val/images/'

# ---------------------------------
# 4. Text supervision
# ---------------------------------
class_text_path = '/mnt/d/Testing VSCode/pest_small-DS/data/texts/class_texts_species_only.json'

# ---------------------------------
# 5. Classes / model head
# ---------------------------------
# Many YOLO-World configs use num_train_classes / num_test_classes
num_train_classes = num_classes
num_test_classes = num_classes

# Make sure bbox head matches your class count
model = dict(
    bbox_head=dict(head_module=dict(num_classes=num_classes)),
    train_cfg=dict(assigner=dict(num_classes=num_classes))
)

# ---------------------------------
# 6. Pipelines
# ---------------------------------
# Reuse the base train pipeline but add text loading before packing.
# The base config already defines train_pipeline and test_pipeline.
train_pipeline = [
    *_base_.train_pipeline[:-1],
    dict(
        type='RandomLoadText',
        num_neg_samples=(num_classes, num_classes),
        max_num_samples=num_classes,
        padding_to_max=True,
        padding_value='',
    ),
    _base_.train_pipeline[-1],
]

train_pipeline_stage2 = [
    *_base_.train_pipeline_stage2[:-1],
    dict(
        type='RandomLoadText',
        num_neg_samples=(num_classes, num_classes),
        max_num_samples=num_classes,
        padding_to_max=True,
        padding_value='',
    ),
    _base_.train_pipeline_stage2[-1],
]

test_pipeline = [
    *_base_.test_pipeline[:-1],
    dict(type='LoadText'),
    _base_.test_pipeline[-1],
]
# ---------------------------------
# 7. Dataset
# ---------------------------------
train_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(
        type='HierYOLOv5CocoDataset',
        data_root=data_root,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_data_prefix),
        metainfo=dict(classes=class_name),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
    ),
    class_text_path=class_text_path,
    pipeline=train_pipeline,
)

train_dataset_stage2 = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(
        type='HierYOLOv5CocoDataset',
        data_root=data_root,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_data_prefix),
        metainfo=dict(classes=class_name),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
    ),
    class_text_path=class_text_path,
    pipeline=train_pipeline_stage2,
)

val_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(
        type='HierYOLOv5CocoDataset',
        data_root=data_root,
        ann_file=val_ann_file,
        data_prefix=dict(img=val_data_prefix),
        metainfo=dict(classes=class_name),
        test_mode=True,
    ),
    class_text_path=class_text_path,
    pipeline=test_pipeline,
)

# ---------------------------------
# 8. Dataloaders
# ---------------------------------
train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    collate_fn=dict(type='yolow_collate'),
    dataset=train_dataset,
)

val_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    dataset=val_dataset,
)

test_dataloader = val_dataloader

# ---------------------------------
# 9. Evaluator
# ---------------------------------
val_evaluator = dict(
    type='mmdet.CocoMetric',
    ann_file=data_root + val_ann_file,
    metric='bbox'
)
test_evaluator = val_evaluator

# ---------------------------------
# 10. Hooks
# ---------------------------------
default_hooks = dict(
    checkpoint=dict(
        interval=save_epoch_intervals,
        max_keep_ckpts=3,
        save_best='auto',
    )
)

# ---------------------------------
# 11. Training schedule
# ---------------------------------
train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=5,
)

# Switch to stage2 pipeline near the end, same as base-style YOLO training
# custom_hooks = [
#     dict(
#         type='PipelineSwitchHook',
#         switch_epoch=max_epochs - close_mosaic_epochs,
#         switch_pipeline=train_pipeline_stage2,
#     )
# ]