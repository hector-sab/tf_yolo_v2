# You Only Look Once V2

This is a reimplementation of the implementation found [here](https://github.com/pmkalshetti/object_detection).

This version does not use tf eager and allow to have the output of multiple images in one batch. However the output is different from the one found in the link above.

## How to use try it

Download the checkpoints from [here](https://correoipn-my.sharepoint.com/:u:/g/personal/hsanchezb1600_alumno_ipn_mx/EdSRX89PwiZCt5AE-cpZRTQB90qs4ZtU44To1VklOETiAA?e=8LHaAo) and place the 'chekpoints' folder at the root of this repository.

Import the model, create an object of Model.
```python
from model import Model
model = Model()
```

Load the image to be used. The expected shape of im by model is [?,im_h,im_w,im_c] where im_h= height,im_w=width,im_c=channels.
```python
im_path = '../test_im/im3.jpg'
im = load_im(im_path)
```

Predict. The model will return 4 objects, the coordinates of two corners of the bounding box of each object detected, the label of each bounding box, the mask that indicates which anchor at any cell contains an object or not, and the prob of the object in each anchor.
```python
coord,lbs,ob_mask,pobj = model.predict(im)
```

If you want to visualize the output image of each of the images do as follow.
```python
from utils import draw_output,plot_ims
ims = draw_output(im,coord,lbs,ob_mask,pobj)
plot_ims(ims)
```

## TODO:
- Implement the training.

- Add NMS.

- Creates the model to train yolo from scratches having the base model, and the two heads (darknet and yolov2).
