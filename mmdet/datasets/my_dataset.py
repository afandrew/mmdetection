from .coco import CocoDataset
from .builder import DATASETS
from .xml_style import XMLDataset

@DATASETS.register_module
class MyDataset(XMLDataset):

    CLASSES = ('AN-2')
