from .coco import CocoDataset
from mmdet.utils import Registry
from .xml_style import XMLDataset

DATASETS = Registry('dataset')
@DATASETS.register_module
class MyDataset(XMLDataset):

    CLASSES = ('AN-2')
