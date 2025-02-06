# fai-m2f-l-ade

## Overview
The models is a [Mask2Former](https://github.com/facebookresearch/Mask2Former) model otimized by [FocoosAI](https://focoos.ai) for the [ADE20K dataset](https://groups.csail.mit.edu/vision/datasets/ADE20K/). It is a semantic segmentation model able to segment 150 classes, comprising both stuff (sky, road, etc.) and thing (dog, cat, car, etc.).


## Benchmark
![Benchmark Comparison](./fai-ade.png)
Note: FPS are computed on NVIDIA T4 using TensorRT and image size 640x640.
<!--
| Model | mIoU | FPS (NVIDIA T4) |
|-------|------|-----------------|
| MobileNetV2+Deeplab | 34.0 | 106 |
| SegFormerB0 | 37.4 | 119 |
| BiSeNetv2-B | 39.2 | 145 |
| DeepLabV3+ (R50) | 45.7 | 30 |
| SegFormerB5 | 49.6 | 27 |
| MaskFormer (R50) | 44.3 | 68 |
| Mask2Former (R50) | 47.2 | 21.5 |
| [fai-m2f-s-ade](models/fai-m2f-s-ade.md) | 41.23 | 189 |
| [fai-m2f-m-ade](models/fai-m2f-m-ade.md) | 45.32 | 127 |
| **fai-m2f-l-ade** | **48.27** | **73** | -->


## Model Details
The model is based on the [Mask2Former](https://github.com/facebookresearch/Mask2Former) architecture. It is a segmentation model that uses a transformer-based encoder-decoder architecture.
Differently from traditional segmentation models (such as [DeepLab](https://arxiv.org/abs/1802.02611)), Mask2Former uses a mask-classification approach, where the prediction is made by a set of segmentation mask with associated class probabilities.

### Neural Network Architecture
The [Mask2Former](https://arxiv.org/abs/2112.01527) FocoosAI implementation optimize the original neural network architecture for improving the model's efficiency and performance. The original model is fully described in this [paper](https://arxiv.org/abs/2112.01527).

Mask2Former is a hybrid model that uses three main components: a *backbone* for extracting features, a *pixel decoder* for upscaling the features, and a *transformer-based decoder* for generating the segmentation output.

![alt text](./mask2former.png)

In this implementation:

- the backbone is a [Resnet-101](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py),that guarantees a good performance while having good efficiency.
- the pixel decoder is a [FPN](https://arxiv.org/abs/1612.03144) getting the features from the stage 2 (1/4 resolution), 3 (1/8 resolution), 4 (1/16 resolution) and 5 (1/32 resolution) of the backbone. Differently from the original paper, for the sake of portability, we removed the deformable attention modules in the pixel decoder, speeding up the inference while only marginally affecting the accuracy.
- the transformer decoder is a lighter version of the original, having only 6 decoder layers (instead of 9) and 100 learnable queries.

### Losses
We use the same losses as the original paper:

- loss_ce: Cross-entropy loss for the classification of the classes
- loss_dice: Dice loss for the segmentation of the classes
- loss_mask: A binary cross-entropy loss applied to the predicted segmentation masks

These losses are applied to each output of the transformer decoder, meaning that we apply it on the output and on each auxiliary output of the 6 transformer decoder layers.
Please refer to the [Mask2Former paper](https://arxiv.org/abs/2112.01527) for more details.

### Output Format
The pre-processed output of the model is set of masks with associated class probabilities. In particular, the output is composed by three tensors:

- class_ids: a tensor of 100 elements containing the class id associated with each mask (such as 1 for wall, 2 for building, etc.)
- scores: a tensor of 100 elements containing the corresponding probability of the class_id
- masks: a tensor of shape (100, H, W) where H and W are the height and width of the input image and the values represent the index of the class_id associated with the pixel

The model does not need NMS (non-maximum suppression) because the output is already a set of masks with associated class probabilities and has been trained to avoid overlapping masks.

After the post-processing, the output is a [Focoos Detections](https://github.com/FocoosAI/focoos/blob/4a317a269cb7758ea71b255faeba654d21182083/focoos/ports.py#L179) object containing the predicted masks with confidence greather than a specific threshold (0.5 by default).


## Classes
The model is pretrained on the [ADE20K dataset](https://groups.csail.mit.edu/vision/datasets/ADE20K/) with 150 classes.

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
      <th>mIoU</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>1</td><td>wall</td><td>77.284</td></tr>
    <tr><td>2</td><td>building</td><td>81.396</td></tr>
    <tr><td>3</td><td>sky</td><td>94.337</td></tr>
    <tr><td>4</td><td>floor</td><td>81.584</td></tr>
    <tr><td>5</td><td>tree</td><td>74.103</td></tr>
    <tr><td>6</td><td>ceiling</td><td>83.073</td></tr>
    <tr><td>7</td><td>road, route</td><td>83.013</td></tr>
    <tr><td>8</td><td>bed</td><td>88.120</td></tr>
    <tr><td>9</td><td>window</td><td>61.048</td></tr>
    <tr><td>10</td><td>grass</td><td>69.099</td></tr>
    <tr><td>11</td><td>cabinet</td><td>56.303</td></tr>
    <tr><td>12</td><td>sidewalk, pavement</td><td>62.300</td></tr>
    <tr><td>13</td><td>person</td><td>82.073</td></tr>
    <tr><td>14</td><td>earth, ground</td><td>35.094</td></tr>
    <tr><td>15</td><td>door</td><td>45.140</td></tr>
    <tr><td>16</td><td>table</td><td>59.436</td></tr>
    <tr><td>17</td><td>mountain, mount</td><td>60.538</td></tr>
    <tr><td>18</td><td>plant</td><td>51.829</td></tr>
    <tr><td>19</td><td>curtain</td><td>71.510</td></tr>
    <tr><td>20</td><td>chair</td><td>56.219</td></tr>
    <tr><td>21</td><td>car</td><td>83.766</td></tr>
    <tr><td>22</td><td>water</td><td>49.028</td></tr>
    <tr><td>23</td><td>painting, picture</td><td>70.214</td></tr>
    <tr><td>24</td><td>sofa</td><td>68.081</td></tr>
    <tr><td>25</td><td>shelf</td><td>35.453</td></tr>
    <tr><td>26</td><td>house</td><td>45.656</td></tr>
    <tr><td>27</td><td>sea</td><td>51.205</td></tr>
    <tr><td>28</td><td>mirror</td><td>61.611</td></tr>
    <tr><td>29</td><td>rug</td><td>64.144</td></tr>
    <tr><td>30</td><td>field</td><td>30.577</td></tr>
    <tr><td>31</td><td>armchair</td><td>45.761</td></tr>
    <tr><td>32</td><td>seat</td><td>61.850</td></tr>
    <tr><td>33</td><td>fence</td><td>40.992</td></tr>
    <tr><td>34</td><td>desk</td><td>41.814</td></tr>
    <tr><td>35</td><td>rock, stone</td><td>47.600</td></tr>
    <tr><td>36</td><td>wardrobe, closet, press</td><td>39.846</td></tr>
    <tr><td>37</td><td>lamp</td><td>64.062</td></tr>
    <tr><td>38</td><td>tub</td><td>74.760</td></tr>
    <tr><td>39</td><td>rail</td><td>24.105</td></tr>
    <tr><td>40</td><td>cushion</td><td>56.811</td></tr>
    <tr><td>41</td><td>base, pedestal, stand</td><td>27.777</td></tr>
    <tr><td>42</td><td>box</td><td>24.670</td></tr>
    <tr><td>43</td><td>column, pillar</td><td>40.094</td></tr>
    <tr><td>44</td><td>signboard, sign</td><td>33.495</td></tr>
    <tr><td>45</td><td>chest of drawers, chest, bureau, dresser</td><td>41.847</td></tr>
    <tr><td>46</td><td>counter</td><td>21.387</td></tr>
    <tr><td>47</td><td>sand</td><td>29.763</td></tr>
    <tr><td>48</td><td>sink</td><td>74.092</td></tr>
    <tr><td>49</td><td>skyscraper</td><td>37.613</td></tr>
    <tr><td>50</td><td>fireplace</td><td>65.037</td></tr>
    <tr><td>51</td><td>refrigerator, icebox</td><td>57.648</td></tr>
    <tr><td>52</td><td>grandstand, covered stand</td><td>46.626</td></tr>
    <tr><td>53</td><td>path</td><td>24.543</td></tr>
    <tr><td>54</td><td>stairs</td><td>28.681</td></tr>
    <tr><td>55</td><td>runway</td><td>73.779</td></tr>
    <tr><td>56</td><td>case, display case, showcase, vitrine</td><td>38.437</td></tr>
    <tr><td>57</td><td>pool table, billiard table, snooker table</td><td>91.825</td></tr>
    <tr><td>58</td><td>pillow</td><td>49.388</td></tr>
    <tr><td>59</td><td>screen door, screen</td><td>59.058</td></tr>
    <tr><td>60</td><td>stairway, staircase</td><td>32.832</td></tr>
    <tr><td>61</td><td>river</td><td>18.597</td></tr>
    <tr><td>62</td><td>bridge, span</td><td>56.011</td></tr>
    <tr><td>63</td><td>bookcase</td><td>28.848</td></tr>
    <tr><td>64</td><td>blind, screen</td><td>43.934</td></tr>
    <tr><td>65</td><td>coffee table</td><td>59.869</td></tr>
    <tr><td>66</td><td>toilet, can, commode, crapper, pot, potty, stool, throne</td><td>86.346</td></tr>
    <tr><td>67</td><td>flower</td><td>38.141</td></tr>
    <tr><td>68</td><td>book</td><td>42.528</td></tr>
    <tr><td>69</td><td>hill</td><td>6.905</td></tr>
    <tr><td>70</td><td>bench</td><td>45.494</td></tr>
    <tr><td>71</td><td>countertop</td><td>49.007</td></tr>
    <tr><td>72</td><td>stove</td><td>73.973</td></tr>
    <tr><td>73</td><td>palm, palm tree</td><td>49.478</td></tr>
    <tr><td>74</td><td>kitchen island</td><td>42.603</td></tr>
    <tr><td>75</td><td>computer</td><td>72.142</td></tr>
    <tr><td>76</td><td>swivel chair</td><td>44.262</td></tr>
    <tr><td>77</td><td>boat</td><td>73.689</td></tr>
    <tr><td>78</td><td>bar</td><td>37.749</td></tr>
    <tr><td>79</td><td>arcade machine</td><td>78.733</td></tr>
    <tr><td>80</td><td>hovel, hut, hutch, shack, shanty</td><td>30.537</td></tr>
    <tr><td>81</td><td>bus</td><td>90.808</td></tr>
    <tr><td>82</td><td>towel</td><td>58.158</td></tr>
    <tr><td>83</td><td>light</td><td>57.444</td></tr>
    <tr><td>84</td><td>truck</td><td>31.745</td></tr>
    <tr><td>85</td><td>tower</td><td>32.058</td></tr>
    <tr><td>86</td><td>chandelier</td><td>67.524</td></tr>
    <tr><td>87</td><td>awning, sunshade, sunblind</td><td>28.566</td></tr>
    <tr><td>88</td><td>street lamp</td><td>30.507</td></tr>
    <tr><td>89</td><td>booth</td><td>39.696</td></tr>
    <tr><td>90</td><td>tv</td><td>76.194</td></tr>
    <tr><td>91</td><td>plane</td><td>50.005</td></tr>
    <tr><td>92</td><td>dirt track</td><td>18.268</td></tr>
    <tr><td>93</td><td>clothes</td><td>37.748</td></tr>
    <tr><td>94</td><td>pole</td><td>23.343</td></tr>
    <tr><td>95</td><td>land, ground, soil</td><td>0.001</td></tr>
    <tr><td>96</td><td>bannister, banister, balustrade, balusters, handrail</td><td>16.222</td></tr>
    <tr><td>97</td><td>escalator, moving staircase, moving stairway</td><td>54.888</td></tr>
    <tr><td>98</td><td>ottoman, pouf, pouffe, puff, hassock</td><td>32.444</td></tr>
    <tr><td>99</td><td>bottle</td><td>22.166</td></tr>
    <tr><td>100</td><td>buffet, counter, sideboard</td><td>48.994</td></tr>
    <tr><td>101</td><td>poster, posting, placard, notice, bill, card</td><td>31.773</td></tr>
    <tr><td>102</td><td>stage</td><td>18.731</td></tr>
    <tr><td>103</td><td>van</td><td>46.747</td></tr>
    <tr><td>104</td><td>ship</td><td>79.937</td></tr>
    <tr><td>105</td><td>fountain</td><td>21.205</td></tr>
    <tr><td>106</td><td>conveyer belt, conveyor belt, conveyer, conveyor, transporter</td><td>62.591</td></tr>
    <tr><td>107</td><td>canopy</td><td>23.719</td></tr>
    <tr><td>108</td><td>washer, automatic washer, washing machine</td><td>66.458</td></tr>
    <tr><td>109</td><td>plaything, toy</td><td>35.377</td></tr>
    <tr><td>110</td><td>pool</td><td>34.297</td></tr>
    <tr><td>111</td><td>stool</td><td>41.199</td></tr>
    <tr><td>112</td><td>barrel, cask</td><td>61.803</td></tr>
    <tr><td>113</td><td>basket, handbasket</td><td>34.313</td></tr>
    <tr><td>114</td><td>falls</td><td>57.149</td></tr>
    <tr><td>115</td><td>tent</td><td>94.077</td></tr>
    <tr><td>116</td><td>bag</td><td>19.126</td></tr>
    <tr><td>117</td><td>minibike, motorbike</td><td>71.207</td></tr>
    <tr><td>118</td><td>cradle</td><td>85.775</td></tr>
    <tr><td>119</td><td>oven</td><td>50.996</td></tr>
    <tr><td>120</td><td>ball</td><td>32.601</td></tr>
    <tr><td>121</td><td>food, solid food</td><td>58.662</td></tr>
    <tr><td>122</td><td>step, stair</td><td>16.474</td></tr>
    <tr><td>123</td><td>tank, storage tank</td><td>37.627</td></tr>
    <tr><td>124</td><td>trade name</td><td>20.788</td></tr>
    <tr><td>125</td><td>microwave</td><td>37.998</td></tr>
    <tr><td>126</td><td>pot</td><td>53.411</td></tr>
    <tr><td>127</td><td>animal</td><td>57.360</td></tr>
    <tr><td>128</td><td>bicycle</td><td>58.772</td></tr>
    <tr><td>129</td><td>lake</td><td>41.597</td></tr>
    <tr><td>130</td><td>dishwasher</td><td>74.543</td></tr>
    <tr><td>131</td><td>screen</td><td>79.757</td></tr>
    <tr><td>132</td><td>blanket, cover</td><td>15.202</td></tr>
    <tr><td>133</td><td>sculpture</td><td>53.537</td></tr>
    <tr><td>134</td><td>hood, exhaust hood</td><td>52.684</td></tr>
    <tr><td>135</td><td>sconce</td><td>48.160</td></tr>
    <tr><td>136</td><td>vase</td><td>45.300</td></tr>
    <tr><td>137</td><td>traffic light</td><td>35.375</td></tr>
    <tr><td>138</td><td>tray</td><td>14.093</td></tr>
    <tr><td>139</td><td>trash can</td><td>30.699</td></tr>
    <tr><td>140</td><td>fan</td><td>56.574</td></tr>
    <tr><td>141</td><td>pier</td><td>10.286</td></tr>
    <tr><td>142</td><td>crt screen</td><td>0.936</td></tr>
    <tr><td>143</td><td>plate</td><td>53.268</td></tr>
    <tr><td>144</td><td>monitor</td><td>9.358</td></tr>
    <tr><td>145</td><td>bulletin board</td><td>29.970</td></tr>
    <tr><td>146</td><td>shower</td><td>8.978</td></tr>
    <tr><td>147</td><td>radiator</td><td>59.763</td></tr>
    <tr><td>148</td><td>glass, drinking glass</td><td>18.246</td></tr>
    <tr><td>149</td><td>clock</td><td>29.088</td></tr>
    <tr><td>150</td><td>flag</td><td>37.727</td></tr>
  </tbody>
</table>

</div>


## What are you waiting? Try it!
```python
from focoos import Focoos
import os

# Initialize the Focoos client with your API key
focoos = Focoos(api_key=os.getenv("FOCOOS_API_KEY"))

# Get the remote model (fai-m2f-l-ade) from Focoos API
model = focoos.get_remote_model("fai-m2f-l-ade")

# Run inference on an image
predictions = model.infer("./image.jpg", threshold=0.5)

# Output the predictions
print(predictions)
```
