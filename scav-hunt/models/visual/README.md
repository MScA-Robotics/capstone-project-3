# Model directory for visual classifiers

## Requirements

In order for the `scaveye.py` to function correctly, a tflite model and a text file of it's associated class labels must be present within a directory in this folder.

* `model.tflite`: compiled tflite model (tpu or non-tpu)
* `labelmap.txt`: text file with `n` class labels for the `model.tflite`, ordered by the classification indicies of the model output, `[0-(n-1)]`.  Text file should only contain `n` class labels on `n` lines with.

The directory structure should like this

    scav-hunt/
    |- models/
    |  |- visual/
    |  |  |- custom_cone_model_edgetpu/
    |  |  |  |- model.tflite
    |  |  |  |- labelmap.txt

## Links to files

***For UChicago Robotics Capstone Use Only***



Cone Model (non-TPU): <https://uchicago.box.com/s/t28v26nak4gbrr4udjy24la5aocbt0a0>
Cone Model (TPU): <https://uchicago.box.com/s/1rg1pxlzv5l84o9ch50ehhbwzcbk6ex0>

Object Model (non-TPU): <https://uchicago.box.com/s/05hy0cpetzd0nq6xwtqw24xwe756taic>
Object Model (TPU): <https://uchicago.box.com/s/qsm34k8lxjeva44grgdujko98vshk9zw>
