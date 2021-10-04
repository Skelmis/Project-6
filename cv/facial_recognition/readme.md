Steps to train a model
----------------------

Delete everything from `./output`

One
---

Add files under `./datasets` where each folder is the name of someone

Also add a folder of 'unknowns'

`python extract_embeddings.py`

Two
---

`python train_model.py`

Three
-----

`FacialRecognition.recognize()`

Taken from:
https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition/