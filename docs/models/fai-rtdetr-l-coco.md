# fai-rtdetr-l-coco

## Overview
The models is the reimplementation of the [RT-DETR](https://github.com/lyuwenyu/RT-DETR) model by [FocoosAI](https://focoos.ai) for the [COCO dataset](https://cocodataset.org/#home). It is a object detection model able to detect 80 thing (dog, cat, car, etc.) classes.


## Benchmark
![Benchmark Comparison](./fai-coco.png)
Note: FPS are computed on NVIDIA T4 using TensorRT and image size 640x640.

## Model Details
The model is based on the [RT-DETR](https://github.com/lyuwenyu/RT-DETR) architecture. It is a object detection model that uses a transformer-based encoder-decoder architecture.

### Neural Network Architecture
This implementation is a reimplementation of the [RT-DETR](https://github.com/lyuwenyu/RT-DETR) model by [FocoosAI](https://focoos.ai). The original model is fully described in this [paper](https://arxiv.org/abs/2304.08069).

RT-DETR is a hybrid model that uses three main components: a *backbone* for extracting features, an *encoder* for upscaling the features, and a *transformer-based decoder* for generating the detection output.
![alt text](./rt-detr.png)

In this implementation:

- the backbone is a [Resnet-50](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py),that guarantees a good performance while having good efficiency.
- the encoder is the Hybrid Encoder, as proposed by the paper, and it is a bi-FPN (bilinear feature pyramid network) that includes a transformer encoder on the smaller feature resolution for improving efficiency.
- The query selection mechanism select the features of the pixels (aka queries) with the highest probability of containing an object and pass them to a transformer decoder head that will generate the final detection output. In this implementation, we select 300 queries and use 6 transformer decoder layers.

### Losses
We use the same losses as the original paper:

- loss_vfl: a variant of the binary cross entropy loss for the classification of the classes that is weighted by the correctness of the predicted bounding boxes IoU.
- loss_bbox: an L1 loss computing the distance between the predicted bounding boxes and the ground truth bounding boxes.
- loss_giou: a loss minimizing the IoU the predicted bounding boxes and the ground truth bounding boxes. For more details look at [GIoU](https://giou.stanford.edu/).

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
    <tr style="text-align: right;">
      <th></th>
      <th>Class</th>
      <th>AP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>person</td>
      <td>63.2</td>
    </tr>
    <tr>
      <td>2</td>
      <td>bicycle</td>
      <td>40.5</td>
    </tr>
    <tr>
      <td>3</td>
      <td>car</td>
      <td>52.3</td>
    </tr>
    <tr>
      <td>4</td>
      <td>motorcycle</td>
      <td>55.0</td>
    </tr>
    <tr>
      <td>5</td>
      <td>airplane</td>
      <td>76.3</td>
    </tr>
    <tr>
      <td>6</td>
      <td>bus</td>
      <td>74.9</td>
    </tr>
    <tr>
      <td>7</td>
      <td>train</td>
      <td>75.0</td>
    </tr>
    <tr>
      <td>8</td>
      <td>truck</td>
      <td>47.9</td>
    </tr>
    <tr>
      <td>9</td>
      <td>boat</td>
      <td>36.6</td>
    </tr>
    <tr>
      <td>10</td>
      <td>traffic light</td>
      <td>32.6</td>
    </tr>
    <tr>
      <td>11</td>
      <td>fire hydrant</td>
      <td>75.5</td>
    </tr>
    <tr>
      <td>12</td>
      <td>stop sign</td>
      <td>71.2</td>
    </tr>
    <tr>
      <td>13</td>
      <td>parking meter</td>
      <td>54.6</td>
    </tr>
    <tr>
      <td>14</td>
      <td>bench</td>
      <td>34.9</td>
    </tr>
    <tr>
      <td>15</td>
      <td>bird</td>
      <td>46.6</td>
    </tr>
    <tr>
      <td>16</td>
      <td>cat</td>
      <td>79.8</td>
    </tr>
    <tr>
      <td>17</td>
      <td>dog</td>
      <td>75.4</td>
    </tr>
    <tr>
      <td>18</td>
      <td>horse</td>
      <td>69.7</td>
    </tr>
    <tr>
      <td>19</td>
      <td>sheep</td>
      <td>63.0</td>
    </tr>
    <tr>
      <td>20</td>
      <td>cow</td>
      <td>68.8</td>
    </tr>
    <tr>
      <td>21</td>
      <td>elephant</td>
      <td>74.1</td>
    </tr>
    <tr>
      <td>22</td>
      <td>bear</td>
      <td>83.2</td>
    </tr>
    <tr>
      <td>23</td>
      <td>zebra</td>
      <td>78.3</td>
    </tr>
    <tr>
      <td>24</td>
      <td>giraffe</td>
      <td>76.9</td>
    </tr>
    <tr>
      <td>25</td>
      <td>backpack</td>
      <td>25.1</td>
    </tr>
    <tr>
      <td>26</td>
      <td>umbrella</td>
      <td>53.8</td>
    </tr>
    <tr>
      <td>27</td>
      <td>handbag</td>
      <td>24.3</td>
    </tr>
    <tr>
      <td>28</td>
      <td>tie</td>
      <td>44.8</td>
    </tr>
    <tr>
      <td>29</td>
      <td>suitcase</td>
      <td>52.6</td>
    </tr>
    <tr>
      <td>30</td>
      <td>frisbee</td>
      <td>75.3</td>
    </tr>
    <tr>
      <td>31</td>
      <td>skis</td>
      <td>37.2</td>
    </tr>
    <tr>
      <td>32</td>
      <td>snowboard</td>
      <td>50.8</td>
    </tr>
    <tr>
      <td>33</td>
      <td>sports ball</td>
      <td>53.9</td>
    </tr>
    <tr>
      <td>34</td>
      <td>kite</td>
      <td>54.8</td>
    </tr>
    <tr>
      <td>35</td>
      <td>baseball bat</td>
      <td>53.2</td>
    </tr>
    <tr>
      <td>36</td>
      <td>baseball glove</td>
      <td>45.3</td>
    </tr>
    <tr>
      <td>37</td>
      <td>skateboard</td>
      <td>63.7</td>
    </tr>
    <tr>
      <td>38</td>
      <td>surfboard</td>
      <td>50.3</td>
    </tr>
    <tr>
      <td>39</td>
      <td>tennis racket</td>
      <td>61.1</td>
    </tr>
    <tr>
      <td>40</td>
      <td>bottle</td>
      <td>48.8</td>
    </tr>
    <tr>
      <td>41</td>
      <td>wine glass</td>
      <td>44.1</td>
    </tr>
    <tr>
      <td>42</td>
      <td>cup</td>
      <td>53.4</td>
    </tr>
    <tr>
      <td>43</td>
      <td>fork</td>
      <td>51.3</td>
    </tr>
    <tr>
      <td>44</td>
      <td>knife</td>
      <td>34.1</td>
    </tr>
    <tr>
      <td>45</td>
      <td>spoon</td>
      <td>33.5</td>
    </tr>
    <tr>
      <td>46</td>
      <td>bowl</td>
      <td>52.1</td>
    </tr>
    <tr>
      <td>47</td>
      <td>banana</td>
      <td>33.0</td>
    </tr>
    <tr>
      <td>48</td>
      <td>apple</td>
      <td>27.1</td>
    </tr>
    <tr>
      <td>49</td>
      <td>sandwich</td>
      <td>48.1</td>
    </tr>
    <tr>
      <td>50</td>
      <td>orange</td>
      <td>37.9</td>
    </tr>
    <tr>
      <td>51</td>
      <td>broccoli</td>
      <td>28.9</td>
    </tr>
    <tr>
      <td>52</td>
      <td>carrot</td>
      <td>28.2</td>
    </tr>
    <tr>
      <td>53</td>
      <td>hot dog</td>
      <td>50.3</td>
    </tr>
    <tr>
      <td>54</td>
      <td>pizza</td>
      <td>62.5</td>
    </tr>
    <tr>
      <td>55</td>
      <td>donut</td>
      <td>62.3</td>
    </tr>
    <tr>
      <td>56</td>
      <td>cake</td>
      <td>47.5</td>
    </tr>
    <tr>
      <td>57</td>
      <td>chair</td>
      <td>41.2</td>
    </tr>
    <tr>
      <td>58</td>
      <td>couch</td>
      <td>57.3</td>
    </tr>
    <tr>
      <td>59</td>
      <td>potted plant</td>
      <td>36.0</td>
    </tr>
    <tr>
      <td>60</td>
      <td>bed</td>
      <td>58.5</td>
    </tr>
    <tr>
      <td>61</td>
      <td>dining table</td>
      <td>39.4</td>
    </tr>
    <tr>
      <td>62</td>
      <td>toilet</td>
      <td>72.6</td>
    </tr>
    <tr>
      <td>63</td>
      <td>tv</td>
      <td>65.8</td>
    </tr>
    <tr>
      <td>64</td>
      <td>laptop</td>
      <td>73.1</td>
    </tr>
    <tr>
      <td>65</td>
      <td>mouse</td>
      <td>67.1</td>
    </tr>
    <tr>
      <td>66</td>
      <td>remote</td>
      <td>48.2</td>
    </tr>
    <tr>
      <td>67</td>
      <td>keyboard</td>
      <td>63.0</td>
    </tr>
    <tr>
      <td>68</td>
      <td>cell phone</td>
      <td>46.0</td>
    </tr>
    <tr>
      <td>69</td>
      <td>microwave</td>
      <td>64.6</td>
    </tr>
    <tr>
      <td>70</td>
      <td>oven</td>
      <td>44.9</td>
    </tr>
    <tr>
      <td>71</td>
      <td>toaster</td>
      <td>50.4</td>
    </tr>
    <tr>
      <td>72</td>
      <td>sink</td>
      <td>45.7</td>
    </tr>
    <tr>
      <td>73</td>
      <td>refrigerator</td>
      <td>69.4</td>
    </tr>
    <tr>
      <td>74</td>
      <td>book</td>
      <td>22.3</td>
    </tr>
    <tr>
      <td>75</td>
      <td>clock</td>
      <td>59.2</td>
    </tr>
    <tr>
      <td>76</td>
      <td>vase</td>
      <td>45.7</td>
    </tr>
    <tr>
      <td>77</td>
      <td>scissors</td>
      <td>42.4</td>
    </tr>
    <tr>
      <td>78</td>
      <td>teddy bear</td>
      <td>59.5</td>
    </tr>
    <tr>
      <td>79</td>
      <td>hair drier</td>
      <td>35.1</td>
    </tr>
    <tr>
      <td>80</td>
      <td>tootdbrush</td>
      <td>42.0</td>
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

# Get the remote model (fai-rtdetr-l-coco) from Focoos API
model = focoos.get_remote_model("fai-rtdetr-l-coco")

# Run inference on an image
predictions = model.infer("./image.jpg", threshold=0.5)

# Output the predictions
print(predictions)
```
