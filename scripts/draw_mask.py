import PIL
import numpy as np
import openpifpaf
import matplotlib.pyplot as plt

image_folder="./images/"
image_name = "Flood_img_demo.jpg"

pil_im = PIL.Image.open(image_folder + image_name).convert('RGB')
im = np.asarray(pil_im)

predictor = openpifpaf.Predictor(checkpoint='shufflenetv2k16-apollo-24')
predictions, gt_anns, image_meta = predictor.pil_image(pil_im)

annotation_painter = openpifpaf.show.AnnotationPainter()
with openpifpaf.show.image_canvas(im) as ax:
    annotation_painter.annotations(ax, predictions)

for car in predictions:
    print(car.data)