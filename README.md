# You Only Look Once V2

This is a reimplementation of the implementation found [here](https://github.com/pmkalshetti/object_detection).

I am not confident using tf eager and keras layers so here is the code that works for me.

## How to use it

Import the model, create an object of Model.
```python
from model import Model
model = Model()
```

Load the image to be used. The expected shape by model is [?,im_h,im_w,im_c] where im_h= height,im_w=width,im_c=channels.
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
-Implement the training.
-Creates the model to train yolo from scratches having the base model, and the two heads (darknet and yolov2)
