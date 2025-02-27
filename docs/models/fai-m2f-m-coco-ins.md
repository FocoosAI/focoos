# fai-m2f-m-coco-ins

## Overview
The models is a [Mask2Former](https://github.com/facebookresearch/Mask2Former) model otimized by [FocoosAI](https://focoos.ai) for the [COCO dataset](https://cocodataset.org/#home). It is an instance segmentation model able to segment 80 thing (dog, cat, car, etc.) classes.

## Benchmark
![Benchmark Comparison](./fai-coco-ins.png)
Note: FPS are computed on NVIDIA T4 using TensorRT and image size 640x640.

## Model Details
The model is based on the [Mask2Former](https://github.com/facebookresearch/Mask2Former) architecture. It is a segmentation model that uses a mask-classification approach and a transformer-based encoder-decoder architecture.

### Neural Network Architecture
The [Mask2Former](https://arxiv.org/abs/2112.01527) FocoosAI implementation optimize the original neural network architecture for improving the model's efficiency and performance. The original model is fully described in this [paper](https://arxiv.org/abs/2112.01527).

Mask2Former is a hybrid model that uses three main components: a *backbone* for extracting features, a *pixel decoder* for upscaling the features, and a *transformer-based decoder* for generating the segmentation output.

![alt text](./mask2former.png)

In this implementation:

- the backbone is [Resnet-101](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py) that show an amazing trade-off between performance and efficiency.
- the pixel decoder is an efficient transformer-augmented [FPN](https://arxiv.org/abs/1612.03144). It gets the features from the stage 2 (1/4 resolution), 3 (1/8 resolution), 4 (1/16 resolution) and 5 (1/32 resolution) of the backbone. It first uses a transformer encoder to process the features at the lowest resolution (stage 5) and then uses a feature pyramid network to upsample the features. This part is different from the original implementation using deformable attention modules.
- the transformer decoder is implemented as in the original paper, having 9 decoder layers and 100 learnable queries.

### Losses
We use the same losses as the original paper:

- loss_ce: Cross-entropy loss for the classification of the classes
- loss_dice: Dice loss for the segmentation of the classes
- loss_mask: A binary cross-entropy loss applied to the predicted segmentation masks

These losses are applied to each output of the transformer decoder, meaning that we apply it on the output and on each auxiliary output of the 3 transformer decoder layers.
Please refer to the [Mask2Former paper](https://arxiv.org/abs/2112.01527) for more details.

### Output Format
The pre-processed output of the model is set of masks with associated class probabilities. In particular, the output is composed by three tensors:

- class_ids: a tensor of 100 elements containing the class id associated with each mask (such as 1 for wall, 2 for building, etc.)
- scores: a tensor of 100 elements containing the corresponding probability of the class_id
- masks: a tensor of shape (100, H, W) where H and W are the height and width of the input image and the values represent the index of the class_id associated with the pixel

The model does not need NMS (non-maximum suppression) because the output is already a set of masks with associated class probabilities and has been trained to avoid overlapping masks.

After the post-processing, the output is a [Focoos Detections](https://github.com/FocoosAI/focoos/blob/4a317a269cb7758ea71b255faeba654d21182083/focoos/ports.py#L179) object containing the predicted masks with confidence greather than a specific threshold (0.5 by default).


## Classes
The model is pretrained on the [COCO dataset](https://cocodataset.org/#home) with 80 classes.

<div class="class-table" markdown>
  <style>
    .class-table {
      max-height: 500px;
      overflow-y: auto;
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
      <th>Segmentation AP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>person</td>
      <td>49.3</td>
    </tr>
    <tr>
      <td>2</td>
      <td>bicycle</td>
      <td>23.5</td>
    </tr>
    <tr>
      <td>3</td>
      <td>car</td>
      <td>41.6</td>
    </tr>
    <tr>
      <td>4</td>
      <td>motorcycle</td>
      <td>40.9</td>
    </tr>
    <tr>
      <td>5</td>
      <td>airplane</td>
      <td>55.1</td>
    </tr>
    <tr>
      <td>6</td>
      <td>bus</td>
      <td>69.0</td>
    </tr>
    <tr>
      <td>7</td>
      <td>train</td>
      <td>70.0</td>
    </tr>
    <tr>
      <td>8</td>
      <td>truck</td>
      <td>43.0</td>
    </tr>
    <tr>
      <td>9</td>
      <td>boat</td>
      <td>26.9</td>
    </tr>
    <tr>
      <td>10</td>
      <td>traffic light</td>
      <td>26.6</td>
    </tr>
    <tr>
      <td>11</td>
      <td>fire hydrant</td>
      <td>68.7</td>
    </tr>
    <tr>
      <td>12</td>
      <td>stop sign</td>
      <td>65.0</td>
    </tr>
    <tr>
      <td>13</td>
      <td>parking meter</td>
      <td>48.6</td>
    </tr>
    <tr>
      <td>14</td>
      <td>bench</td>
      <td>21.9</td>
    </tr>
    <tr>
      <td>15</td>
      <td>bird</td>
      <td>34.6</td>
    </tr>
    <tr>
      <td>16</td>
      <td>cat</td>
      <td>79.6</td>
    </tr>
    <tr>
      <td>17</td>
      <td>dog</td>
      <td>69.6</td>
    </tr>
    <tr>
      <td>18</td>
      <td>horse</td>
      <td>51.4</td>
    </tr>
    <tr>
      <td>19</td>
      <td>sheep</td>
      <td>54.1</td>
    </tr>
    <tr>
      <td>20</td>
      <td>cow</td>
      <td>51.9</td>
    </tr>
    <tr>
      <td>21</td>
      <td>elephant</td>
      <td>64.3</td>
    </tr>
    <tr>
      <td>22</td>
      <td>bear</td>
      <td>80.5</td>
    </tr>
    <tr>
      <td>23</td>
      <td>zebra</td>
      <td>66.0</td>
    </tr>
    <tr>
      <td>24</td>
      <td>giraffe</td>
      <td>61.1</td>
    </tr>
    <tr>
      <td>25</td>
      <td>backpack</td>
      <td>22.8</td>
    </tr>
    <tr>
      <td>26</td>
      <td>umbrella</td>
      <td>52.3</td>
    </tr>
    <tr>
      <td>27</td>
      <td>handbag</td>
      <td>21.7</td>
    </tr>
    <tr>
      <td>28</td>
      <td>tie</td>
      <td>35.3</td>
    </tr>
    <tr>
      <td>29</td>
      <td>suitcase</td>
      <td>46.2</td>
    </tr>
    <tr>
      <td>30</td>
      <td>frisbee</td>
      <td>65.4</td>
    </tr>
    <tr>
      <td>31</td>
      <td>skis</td>
      <td>8.4</td>
    </tr>
    <tr>
      <td>32</td>
      <td>snowboard</td>
      <td>27.0</td>
    </tr>
    <tr>
      <td>33</td>
      <td>sports ball</td>
      <td>42.1</td>
    </tr>
    <tr>
      <td>34</td>
      <td>kite</td>
      <td>32.5</td>
    </tr>
    <tr>
      <td>35</td>
      <td>baseball bat</td>
      <td>31.0</td>
    </tr>
    <tr>
      <td>36</td>
      <td>baseball glove</td>
      <td>39.5</td>
    </tr>
    <tr>
      <td>37</td>
      <td>skateboard</td>
      <td>40.0</td>
    </tr>
    <tr>
      <td>38</td>
      <td>surfboard</td>
      <td>37.6</td>
    </tr>
    <tr>
      <td>39</td>
      <td>tennis racket</td>
      <td>59.1</td>
    </tr>
    <tr>
      <td>40</td>
      <td>bottle</td>
      <td>38.9</td>
    </tr>
    <tr>
      <td>41</td>
      <td>wine glass</td>
      <td>38.3</td>
    </tr>
    <tr>
      <td>42</td>
      <td>cup</td>
      <td>48.0</td>
    </tr>
    <tr>
      <td>43</td>
      <td>fork</td>
      <td>24.5</td>
    </tr>
    <tr>
      <td>44</td>
      <td>knife</td>
      <td>17.8</td>
    </tr>
    <tr>
      <td>45</td>
      <td>spoon</td>
      <td>19.6</td>
    </tr>
    <tr>
      <td>46</td>
      <td>bowl</td>
      <td>43.0</td>
    </tr>
    <tr>
      <td>47</td>
      <td>banana</td>
      <td>22.6</td>
    </tr>
    <tr>
      <td>48</td>
      <td>apple</td>
      <td>22.3</td>
    </tr>
    <tr>
      <td>49</td>
      <td>sandwich</td>
      <td>46.3</td>
    </tr>
    <tr>
      <td>50</td>
      <td>orange</td>
      <td>31.6</td>
    </tr>
    <tr>
      <td>51</td>
      <td>broccoli</td>
      <td>24.9</td>
    </tr>
    <tr>
      <td>52</td>
      <td>carrot</td>
      <td>20.9</td>
    </tr>
    <tr>
      <td>53</td>
      <td>hot dog</td>
      <td>42.3</td>
    </tr>
    <tr>
      <td>54</td>
      <td>pizza</td>
      <td>53.9</td>
    </tr>
    <tr>
      <td>55</td>
      <td>donut</td>
      <td>52.8</td>
    </tr>
    <tr>
      <td>56</td>
      <td>cake</td>
      <td>44.0</td>
    </tr>
    <tr>
      <td>57</td>
      <td>chair</td>
      <td>25.8</td>
    </tr>
    <tr>
      <td>58</td>
      <td>couch</td>
      <td>44.9</td>
    </tr>
    <tr>
      <td>59</td>
      <td>potted plant</td>
      <td>27.4</td>
    </tr>
    <tr>
      <td>60</td>
      <td>bed</td>
      <td>46.8</td>
    </tr>
    <tr>
      <td>61</td>
      <td>dining table</td>
      <td>22.1</td>
    </tr>
    <tr>
      <td>62</td>
      <td>toilet</td>
      <td>67.0</td>
    </tr>
    <tr>
      <td>63</td>
      <td>tv</td>
      <td>65.8</td>
    </tr>
    <tr>
      <td>64</td>
      <td>laptop</td>
      <td>67.9</td>
    </tr>
    <tr>
      <td>65</td>
      <td>mouse</td>
      <td>59.2</td>
    </tr>
    <tr>
      <td>66</td>
      <td>remote</td>
      <td>36.8</td>
    </tr>
    <tr>
      <td>67</td>
      <td>keyboard</td>
      <td>53.2</td>
    </tr>
    <tr>
      <td>68</td>
      <td>cell phone</td>
      <td>39.8</td>
    </tr>
    <tr>
      <td>69</td>
      <td>microwave</td>
      <td>63.4</td>
    </tr>
    <tr>
      <td>70</td>
      <td>oven</td>
      <td>36.4</td>
    </tr>
    <tr>
      <td>71</td>
      <td>toaster</td>
      <td>47.0</td>
    </tr>
    <tr>
      <td>72</td>
      <td>sink</td>
      <td>41.1</td>
    </tr>
    <tr>
      <td>73</td>
      <td>refrigerator</td>
      <td>64.6</td>
    </tr>
    <tr>
      <td>74</td>
      <td>book</td>
      <td>12.3</td>
    </tr>
    <tr>
      <td>75</td>
      <td>clock</td>
      <td>53.0</td>
    </tr>
    <tr>
      <td>76</td>
      <td>vase</td>
      <td>38.9</td>
    </tr>
    <tr>
      <td>77</td>
      <td>scissors</td>
      <td>28.7</td>
    </tr>
    <tr>
      <td>78</td>
      <td>teddy bear</td>
      <td>53.8</td>
    </tr>
    <tr>
      <td>79</td>
      <td>hair drier</td>
      <td>14.1</td>
    </tr>
    <tr>
      <td>80</td>
      <td>toothbrush</td>
      <td>22.2</td>
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

# Get the remote model (fai-m2f-m-coco-ins) from Focoos API
model = focoos.get_remote_model("fai-m2f-m-coco-ins")

# Run inference on an image
predictions = model.infer("./image.jpg", threshold=0.5)

# Output the predictions
print(predictions)
```
