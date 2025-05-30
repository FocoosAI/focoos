# fai-detr-s-coco

## Overview
The models is a [RT-DETR](https://github.com/lyuwenyu/RT-DETR) model otimized by [FocoosAI](https://focoos.ai) for the [COCO dataset](https://cocodataset.org/#home). It is a object detection model able to detect 80 thing (dog, cat, car, etc.) classes.


## Benchmark
![Benchmark Comparison](./fai-coco.png)
Note: FPS are computed on NVIDIA T4 using TensorRT and image size 640x640.

## Model Details
The model is based on the [RT-DETR](https://github.com/lyuwenyu/RT-DETR) architecture. It is a object detection model that uses a transformer-based encoder-decoder architecture.

### Neural Network Architecture
The [RT-DETR](https://github.com/lyuwenyu/RT-DETR) FocoosAI implementation optimize the original neural network architecture for improving the model's efficiency and performance. The original model is fully described in this [paper](https://arxiv.org/abs/2304.08069).

RT-DETR is a hybrid model that uses three main components: a *backbone* for extracting features, an *encoder* for upscaling the features, and a *transformer-based decoder* for generating the detection output.

![alt text](./rt-detr.png)

In this implementation:

- the backbone is [STDC-2](https://github.com/MichaelFan01/STDC-Seg) that show an amazing trade-off between performance and efficiency.
- the encoder is a bi-FPN (bilinear feature pyramid network). With respect to the original paper, we removed the attention modules in the encoder and we reduce the internal features dimension, speeding up the inference while only marginally affecting the accuracy.
- the transformer decoder is a lighter version of the original, having only 3 decoder layers, instead of 6, and we select 300 queries.

### Losses
We use the same losses as the original paper:

- loss_vfl: a variant of the binary cross entropy loss for the classification of the classes that is weighted by the correctness of the predicted bounding boxes IoU.
- loss_bbox: an L1 loss computing the distance between the predicted bounding boxes and the ground truth bounding boxes.
- loss_giou: a loss minimizing the IoU the predicted bounding boxes and the ground truth bounding boxes. for more details look here: [GIoU](https://giou.stanford.edu/).

These losses are applied to each output of the transformer decoder, meaning that we apply it on the output and on each auxiliary output of the transformer decoder layers.
Please refer to the [RT-DETR paper](https://arxiv.org/abs/2304.08069) for more details.

### Output Format
The pre-processed output of the model is set of bounding boxes with associated class probabilities. In particular, the output is composed by three tensors:

- class_ids: a tensor of 300 elements containing the class id associated with each bounding box (such as 1 for wall, 2 for building, etc.)
- scores: a tensor of 300 elements containing the corresponding probability of the class_id
- boxes: a tensor of shape (300, 4) where the values represent the coordinates of the bounding boxes in the format [x1, y1, x2, y2]

The model does not need NMS (non-maximum suppression) because the output is already a set of bounding boxes with associated class probabilities and has been trained to avoid overlaps.

After the post-processing, the output is a the output is a [Focoos Detections](https://github.com/FocoosAI/focoos/blob/4a317a269cb7758ea71b255faeba654d21182083/focoos/ports.py#L179) object containing the predicted bounding boxes with confidence greather than a specific threshold (0.5 by default).


## Classes
The model is pretrained on the [COCO dataset](https://cocodataset.org/#home) with 80 classes.

<div class="class-table" markdown>
  <style>
    .class-table {
      max-height: 500px;
      overflow-y: auto;
      /* border: 1px solid #ccc; */
      /* border-radius: 4px; */
      padding: 1rem;
      margin: 1rem 0;
      background: rgba(0,0,0,0.05);
      width: 95%;
      margin-left: auto;
      margin-right: auto;
    }
    .class-table table {
      width: 100%;
    }
    .class-table thead {
      position: sticky;
      top: 0;
      background: #2b2b2b;
      z-index: 1;
    }
  </style>
<table>
  <thead>
    <tr>
      <th>Class ID</th>
      <th>Class Name</th>
      <th>AP</th>
    </tr>
  </thead>
  <tbody>
  <tr>
      <td>1</td>
      <td>person</td>
      <td>54.7</td>
    </tr>
    <tr>
      <td>2</td>
      <td>bicycle</td>
      <td>29.1</td>
    </tr>
    <tr>
      <td>3</td>
      <td>car</td>
      <td>41.4</td>
    </tr>
    <tr>
      <td>4</td>
      <td>motorcycle</td>
      <td>44.9</td>
    </tr>
    <tr>
      <td>5</td>
      <td>airplane</td>
      <td>71.4</td>
    </tr>
    <tr>
      <td>6</td>
      <td>bus</td>
      <td>67.8</td>
    </tr>
    <tr>
      <td>7</td>
      <td>train</td>
      <td>68.9</td>
    </tr>
    <tr>
      <td>8</td>
      <td>truck</td>
      <td>36.4</td>
    </tr>
    <tr>
      <td>9</td>
      <td>boat</td>
      <td>26.8</td>
    </tr>
    <tr>
      <td>10</td>
      <td>traffic light</td>
      <td>25.0</td>
    </tr>
    <tr>
      <td>11</td>
      <td>fire hydrant</td>
      <td>66.0</td>
    </tr>
    <tr>
      <td>12</td>
      <td>stop sign</td>
      <td>62.2</td>
    </tr>
    <tr>
      <td>13</td>
      <td>parking meter</td>
      <td>46.1</td>
    </tr>
    <tr>
      <td>14</td>
      <td>bench</td>
      <td>25.2</td>
    </tr>
    <tr>
      <td>15</td>
      <td>bird</td>
      <td>36.5</td>
    </tr>
    <tr>
      <td>16</td>
      <td>cat</td>
      <td>72.6</td>
    </tr>
    <tr>
      <td>17</td>
      <td>dog</td>
      <td>68.5</td>
    </tr>
    <tr>
      <td>18</td>
      <td>horse</td>
      <td>57.9</td>
    </tr>
    <tr>
      <td>19</td>
      <td>sheep</td>
      <td>54.1</td>
    </tr>
    <tr>
      <td>20</td>
      <td>cow</td>
      <td>56.6</td>
    </tr>
    <tr>
      <td>21</td>
      <td>elephant</td>
      <td>66.2</td>
    </tr>
    <tr>
      <td>22</td>
      <td>bear</td>
      <td>78.3</td>
    </tr>
    <tr>
      <td>23</td>
      <td>zebra</td>
      <td>70.0</td>
    </tr>
    <tr>
      <td>24</td>
      <td>giraffe</td>
      <td>70.0</td>
    </tr>
    <tr>
      <td>25</td>
      <td>backpack</td>
      <td>14.9</td>
    </tr>
    <tr>
      <td>26</td>
      <td>umbrella</td>
      <td>39.9</td>
    </tr>
    <tr>
      <td>27</td>
      <td>handbag</td>
      <td>13.2</td>
    </tr>
    <tr>
      <td>28</td>
      <td>tie</td>
      <td>32.6</td>
    </tr>
    <tr>
      <td>29</td>
      <td>suitcase</td>
      <td>41.2</td>
    </tr>
    <tr>
      <td>30</td>
      <td>frisbee</td>
      <td>66.3</td>
    </tr>
    <tr>
      <td>31</td>
      <td>skis</td>
      <td>24.9</td>
    </tr>
    <tr>
      <td>32</td>
      <td>snowboard</td>
      <td>31.6</td>
    </tr>
    <tr>
      <td>33</td>
      <td>sports ball</td>
      <td>44.8</td>
    </tr>
    <tr>
      <td>34</td>
      <td>kite</td>
      <td>45.1</td>
    </tr>
    <tr>
      <td>35</td>
      <td>baseball bat</td>
      <td>29.7</td>
    </tr>
    <tr>
      <td>36</td>
      <td>baseball glove</td>
      <td>35.2</td>
    </tr>
    <tr>
      <td>37</td>
      <td>skateboard</td>
      <td>54.5</td>
    </tr>
    <tr>
      <td>38</td>
      <td>surfboard</td>
      <td>39.9</td>
    </tr>
    <tr>
      <td>39</td>
      <td>tennis racket</td>
      <td>46.1</td>
    </tr>
    <tr>
      <td>40</td>
      <td>bottle</td>
      <td>35.8</td>
    </tr>
    <tr>
      <td>41</td>
      <td>wine glass</td>
      <td>32.6</td>
    </tr>
    <tr>
      <td>42</td>
      <td>cup</td>
      <td>41.1</td>
    </tr>
    <tr>
      <td>43</td>
      <td>fork</td>
      <td>35.5</td>
    </tr>
    <tr>
      <td>44</td>
      <td>knife</td>
      <td>18.9</td>
    </tr>
    <tr>
      <td>45</td>
      <td>spoon</td>
      <td>18.0</td>
    </tr>
    <tr>
      <td>46</td>
      <td>bowl</td>
      <td>42.2</td>
    </tr>
    <tr>
      <td>47</td>
      <td>banana</td>
      <td>24.6</td>
    </tr>
    <tr>
      <td>48</td>
      <td>apple</td>
      <td>18.6</td>
    </tr>
    <tr>
      <td>49</td>
      <td>sandwich</td>
      <td>41.6</td>
    </tr>
    <tr>
      <td>50</td>
      <td>orange</td>
      <td>33.1</td>
    </tr>
    <tr>
      <td>51</td>
      <td>broccoli</td>
      <td>22.4</td>
    </tr>
    <tr>
      <td>52</td>
      <td>carrot</td>
      <td>22.2</td>
    </tr>
    <tr>
      <td>53</td>
      <td>hot dog</td>
      <td>37.6</td>
    </tr>
    <tr>
      <td>54</td>
      <td>pizza</td>
      <td>55.2</td>
    </tr>
    <tr>
      <td>55</td>
      <td>donut</td>
      <td>48.0</td>
    </tr>
    <tr>
      <td>56</td>
      <td>cake</td>
      <td>36.7</td>
    </tr>
    <tr>
      <td>57</td>
      <td>chair</td>
      <td>28.4</td>
    </tr>
    <tr>
      <td>58</td>
      <td>couch</td>
      <td>47.8</td>
    </tr>
    <tr>
      <td>59</td>
      <td>potted plant</td>
      <td>26.8</td>
    </tr>
    <tr>
      <td>60</td>
      <td>bed</td>
      <td>49.0</td>
    </tr>
    <tr>
      <td>61</td>
      <td>dining table</td>
      <td>30.5</td>
    </tr>
    <tr>
      <td>62</td>
      <td>toilet</td>
      <td>60.1</td>
    </tr>
    <tr>
      <td>63</td>
      <td>tv</td>
      <td>57.2</td>
    </tr>
    <tr>
      <td>64</td>
      <td>laptop</td>
      <td>59.6</td>
    </tr>
    <tr>
      <td>65</td>
      <td>mouse</td>
      <td>62.3</td>
    </tr>
    <tr>
      <td>66</td>
      <td>remote</td>
      <td>27.7</td>
    </tr>
    <tr>
      <td>67</td>
      <td>keyboard</td>
      <td>53.8</td>
    </tr>
    <tr>
      <td>68</td>
      <td>cell phone</td>
      <td>33.2</td>
    </tr>
    <tr>
      <td>69</td>
      <td>microwave</td>
      <td>60.7</td>
    </tr>
    <tr>
      <td>70</td>
      <td>oven</td>
      <td>38.8</td>
    </tr>
    <tr>
      <td>71</td>
      <td>toaster</td>
      <td>41.9</td>
    </tr>
    <tr>
      <td>72</td>
      <td>sink</td>
      <td>37.0</td>
    </tr>
    <tr>
      <td>73</td>
      <td>refrigerator</td>
      <td>57.6</td>
    </tr>
    <tr>
      <td>74</td>
      <td>book</td>
      <td>13.8</td>
    </tr>
    <tr>
      <td>75</td>
      <td>clock</td>
      <td>50.3</td>
    </tr>
    <tr>
      <td>76</td>
      <td>vase</td>
      <td>35.5</td>
    </tr>
    <tr>
      <td>77</td>
      <td>scissors</td>
      <td>31.8</td>
    </tr>
    <tr>
      <td>78</td>
      <td>teddy bear</td>
      <td>44.7</td>
    </tr>
    <tr>
      <td>79</td>
      <td>hair drier</td>
      <td>10.3</td>
    </tr>
    <tr>
      <td>80</td>
      <td>toothbrush</td>
      <td>26.8</td>
    </tr>
  </tbody>
</table>

</div>


## What are you waiting? Try it!
```python
from focoos import Focoos
import os

# Initialize the Focoos client with your API key
focoos = Focoos(api_key=os.getenv("FOCOOS_API_KEY"))

# Get the remote model (fai-detr-s-coco) from Focoos API
model = focoos.get_remote_model("fai-detr-s-coco")

# Run inference on an image
predictions = model.infer("./image.jpg", threshold=0.5)

# Output the predictions
print(predictions)
```
