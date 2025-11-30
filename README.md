# cisc-473-group-14

Run train.py to train a network.

~~~
pyhton3 train.py NUM_EPOCHS NUM_STROKES
~~~

This automatically saves it to a file at models/[NUM_EPOCHS]_[NUM_STROKES]

To run inference, use:

~~~
python3 test.py MODEL_FILEPATH IMAGE_FOLDER_FILEPATH NUM_STROKES
~~~

Note that, as per pytorch standard, the image folder filepath should contain a subfolder which contains the images. The folder at IMAGE_FOLDER_FILEPATH should *not* contain the images themselves.