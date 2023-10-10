import openpifpaf
import torch
import numpy as np
import PIL
import matplotlib.pyplot as plt

pil_im = PIL.Image.open("./images/Flood_1409.jpg").convert('RGB')
im = np.asarray(pil_im)

predictor = openpifpaf.Predictor(checkpoint='shufflenetv2k16-apollo-66')
predictions, gt_anns, image_meta = predictor.pil_image(pil_im)
# annotation_painter = openpifpaf.show.painters.KeypointPainter(show_box = True)

annotation_painter = openpifpaf.show.AnnotationPainter()
with openpifpaf.show.image_canvas(im) as ax:
    annotation_painter.annotations(ax, predictions)