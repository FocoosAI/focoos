pass
# from typing import Optional

# import numpy as np
# from PIL import Image
# from supervisely import Annotation, Label, ObjClass, ProjectMeta
# from supervisely.geometry.bitmap import Bitmap
# from supervisely.geometry.polygon import Polygon

# from anyma.utils.helpers import time_track


# def sly_ann_to_bitmap_mask(ann: Annotation, out_path: str, sly_meta: ProjectMeta, colors: Optional[int] = 256):
#     classes = [obj_cls.name for obj_cls in sly_meta.obj_classes]
#     if isinstance(ann.labels[0].geometry, Polygon):
#         # convert polygon to bitmap
#         mapping = {}
#         for obj_class in sly_meta.obj_classes:
#             new_obj_class = ObjClass(obj_class.name, Bitmap)
#             mapping[obj_class] = new_obj_class
#         ann = ann.to_nonoverlapping_masks(mapping)

#     mask = np.zeros((ann.img_size[0], ann.img_size[1]), dtype=np.uint8)
#     for label in ann.labels:
#         label.geometry.draw(mask, classes.index(label.obj_class.name))
#     im = Image.fromarray(mask.astype(np.uint8))
#     # im = im.convert("P", palette=palette, colors=colors)
#     im.save(out_path)
