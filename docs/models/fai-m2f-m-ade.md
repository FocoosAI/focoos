# fai-m2f-m-ade

## Overview
The models is a [Mask2Former](https://github.com/facebookresearch/Mask2Former) model otimized by [FocoosAI](https://focoos.ai) for the [ADE20K dataset](https://groups.csail.mit.edu/vision/datasets/ADE20K/). It is a semantic segmentation model able to segment 150 classes, comprising both stuff (sky, road, etc.) and thing (dog, cat, car, etc.).


## Benchmark
![Benchmark Comparison](./fai-ade.png)
Note: FPS are computed on NVIDIA T4 using TensorRT and image size 640x640.

## Model Details
The model is based on the [Mask2Former](https://github.com/facebookresearch/Mask2Former) architecture. It is a segmentation model that uses a transformer-based encoder-decoder architecture.
Differently from traditional segmentation models (such as [DeepLab](https://arxiv.org/abs/1802.02611)), Mask2Former uses a mask-classification approach, where the prediction is made by a set of segmentation mask with associated class probabilities.

### Neural Network Architecture
The [Mask2Former](https://arxiv.org/abs/2112.01527) FocoosAI implementation optimize the original neural network architecture for improving the model's efficiency and performance. The original model is fully described in this [paper](https://arxiv.org/abs/2112.01527).

Mask2Former is a hybrid model that uses three main components: a *backbone* for extracting features, a *pixel decoder* for upscaling the features, and a *transformer-based decoder* for generating the segmentation output.
![alt text](./mask2former.png)

In this implementation:

- the backbone is [STDC-2](https://github.com/MichaelFan01/STDC-Seg) that show an amazing trade-off between performance and efficiency.
- the pixel decoder is a [FPN](https://arxiv.org/abs/1612.03144) getting the features from the stage 2 (1/4 resolution), 3 (1/8 resolution), 4 (1/16 resolution) and 5 (1/32 resolution) of the backbone. Differently from the original paper, for the sake of portability, we removed the deformable attention modules in the pixel decoder, speeding up the inference while only marginally affecting the accuracy.
- the transformer decoder is a lighter version of the original, having only 3 decoder layers (instead of 9) and 100 learnable queries.

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
    <tr style="text-align: right;">
      <th></th>
      <th>Class</th>
      <th>mIoU</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>wall</td>
      <td>75.369549</td>
    </tr>
    <tr>
      <td>2</td>
      <td>building</td>
      <td>79.835995</td>
    </tr>
    <tr>
      <td>3</td>
      <td>sky</td>
      <td>94.176995</td>
    </tr>
    <tr>
      <td>4</td>
      <td>floor</td>
      <td>79.620841</td>
    </tr>
    <tr>
      <td>5</td>
      <td>tree</td>
      <td>73.204506</td>
    </tr>
    <tr>
      <td>6</td>
      <td>ceiling</td>
      <td>82.303035</td>
    </tr>
    <tr>
      <td>7</td>
      <td>road, route</td>
      <td>80.822591</td>
    </tr>
    <tr>
      <td>8</td>
      <td>bed</td>
      <td>87.573840</td>
    </tr>
    <tr>
      <td>9</td>
      <td>window</td>
      <td>57.452584</td>
    </tr>
    <tr>
      <td>10</td>
      <td>grass</td>
      <td>70.099493</td>
    </tr>
    <tr>
      <td>11</td>
      <td>cabinet</td>
      <td>56.903790</td>
    </tr>
    <tr>
      <td>12</td>
      <td>sidewalk, pavement</td>
      <td>62.247267</td>
    </tr>
    <tr>
      <td>13</td>
      <td>person</td>
      <td>79.460606</td>
    </tr>
    <tr>
      <td>14</td>
      <td>earth, ground</td>
      <td>38.537802</td>
    </tr>
    <tr>
      <td>15</td>
      <td>door</td>
      <td>43.930878</td>
    </tr>
    <tr>
      <td>16</td>
      <td>table</td>
      <td>56.753292</td>
    </tr>
    <tr>
      <td>17</td>
      <td>mountain, mount</td>
      <td>61.160462</td>
    </tr>
    <tr>
      <td>18</td>
      <td>plant</td>
      <td>48.995487</td>
    </tr>
    <tr>
      <td>19</td>
      <td>curtain</td>
      <td>71.951930</td>
    </tr>
    <tr>
      <td>20</td>
      <td>chair</td>
      <td>52.852125</td>
    </tr>
    <tr>
      <td>21</td>
      <td>car</td>
      <td>80.725703</td>
    </tr>
    <tr>
      <td>22</td>
      <td>water</td>
      <td>51.233498</td>
    </tr>
    <tr>
      <td>23</td>
      <td>painting, picture</td>
      <td>66.989493</td>
    </tr>
    <tr>
      <td>24</td>
      <td>sofa</td>
      <td>58.103663</td>
    </tr>
    <tr>
      <td>25</td>
      <td>shelf</td>
      <td>34.979205</td>
    </tr>
    <tr>
      <td>26</td>
      <td>house</td>
      <td>36.828611</td>
    </tr>
    <tr>
      <td>27</td>
      <td>sea</td>
      <td>51.219096</td>
    </tr>
    <tr>
      <td>28</td>
      <td>mirror</td>
      <td>58.572852</td>
    </tr>
    <tr>
      <td>29</td>
      <td>rug</td>
      <td>54.897799</td>
    </tr>
    <tr>
      <td>30</td>
      <td>field</td>
      <td>29.053876</td>
    </tr>
    <tr>
      <td>31</td>
      <td>armchair</td>
      <td>39.565663</td>
    </tr>
    <tr>
      <td>32</td>
      <td>seat</td>
      <td>53.113668</td>
    </tr>
    <tr>
      <td>33</td>
      <td>fence</td>
      <td>41.113128</td>
    </tr>
    <tr>
      <td>34</td>
      <td>desk</td>
      <td>37.930189</td>
    </tr>
    <tr>
      <td>35</td>
      <td>rock, stone</td>
      <td>44.940982</td>
    </tr>
    <tr>
      <td>36</td>
      <td>wardrobe, closet, press</td>
      <td>39.897858</td>
    </tr>
    <tr>
      <td>37</td>
      <td>lamp</td>
      <td>60.921356</td>
    </tr>
    <tr>
      <td>38</td>
      <td>tub</td>
      <td>78.041637</td>
    </tr>
    <tr>
      <td>39</td>
      <td>rail</td>
      <td>31.893878</td>
    </tr>
    <tr>
      <td>40</td>
      <td>cushion</td>
      <td>53.029316</td>
    </tr>
    <tr>
      <td>41</td>
      <td>base, pedestal, stand</td>
      <td>20.233620</td>
    </tr>
    <tr>
      <td>42</td>
      <td>box</td>
      <td>18.276924</td>
    </tr>
    <tr>
      <td>43</td>
      <td>column, pillar</td>
      <td>42.655306</td>
    </tr>
    <tr>
      <td>44</td>
      <td>signboard, sign</td>
      <td>35.959448</td>
    </tr>
    <tr>
      <td>45</td>
      <td>chest of drawers, chest, bureau, dresser</td>
      <td>36.521600</td>
    </tr>
    <tr>
      <td>46</td>
      <td>counter</td>
      <td>29.353667</td>
    </tr>
    <tr>
      <td>47</td>
      <td>sand</td>
      <td>38.729599</td>
    </tr>
    <tr>
      <td>48</td>
      <td>sink</td>
      <td>72.303141</td>
    </tr>
    <tr>
      <td>49</td>
      <td>skyscraper</td>
      <td>44.122387</td>
    </tr>
    <tr>
      <td>50</td>
      <td>fireplace</td>
      <td>66.614683</td>
    </tr>
    <tr>
      <td>51</td>
      <td>refrigerator, icebox</td>
      <td>72.137179</td>
    </tr>
    <tr>
      <td>52</td>
      <td>grandstand, covered stand</td>
      <td>29.061628</td>
    </tr>
    <tr>
      <td>53</td>
      <td>path</td>
      <td>26.629478</td>
    </tr>
    <tr>
      <td>54</td>
      <td>stairs</td>
      <td>31.833328</td>
    </tr>
    <tr>
      <td>55</td>
      <td>runway</td>
      <td>76.017706</td>
    </tr>
    <tr>
      <td>56</td>
      <td>case, display case, showcase, vitrine</td>
      <td>37.452627</td>
    </tr>
    <tr>
      <td>57</td>
      <td>pool table, billiard table, snooker table</td>
      <td>93.246039</td>
    </tr>
    <tr>
      <td>58</td>
      <td>pillow</td>
      <td>54.689591</td>
    </tr>
    <tr>
      <td>59</td>
      <td>screen door, screen</td>
      <td>58.096890</td>
    </tr>
    <tr>
      <td>60</td>
      <td>stairway, staircase</td>
      <td>29.962829</td>
    </tr>
    <tr>
      <td>61</td>
      <td>river</td>
      <td>15.010211</td>
    </tr>
    <tr>
      <td>62</td>
      <td>bridge, span</td>
      <td>66.617580</td>
    </tr>
    <tr>
      <td>63</td>
      <td>bookcase</td>
      <td>31.383789</td>
    </tr>
    <tr>
      <td>64</td>
      <td>blind, screen</td>
      <td>39.221180</td>
    </tr>
    <tr>
      <td>65</td>
      <td>coffee table</td>
      <td>63.300795</td>
    </tr>
    <tr>
      <td>66</td>
      <td>toilet, can, commode, crapper, pot, potty, stool, throne</td>
      <td>84.038177</td>
    </tr>
    <tr>
      <td>67</td>
      <td>flower</td>
      <td>35.994798</td>
    </tr>
    <tr>
      <td>68</td>
      <td>book</td>
      <td>43.252042</td>
    </tr>
    <tr>
      <td>69</td>
      <td>hill</td>
      <td>6.240850</td>
    </tr>
    <tr>
      <td>70</td>
      <td>bench</td>
      <td>35.007473</td>
    </tr>
    <tr>
      <td>71</td>
      <td>countertop</td>
      <td>56.592858</td>
    </tr>
    <tr>
      <td>72</td>
      <td>stove</td>
      <td>74.866261</td>
    </tr>
    <tr>
      <td>73</td>
      <td>palm, palm tree</td>
      <td>49.092486</td>
    </tr>
    <tr>
      <td>74</td>
      <td>kitchen island</td>
      <td>32.353614</td>
    </tr>
    <tr>
      <td>75</td>
      <td>computer</td>
      <td>57.673329</td>
    </tr>
    <tr>
      <td>76</td>
      <td>swivel chair</td>
      <td>43.202283</td>
    </tr>
    <tr>
      <td>77</td>
      <td>boat</td>
      <td>48.170742</td>
    </tr>
    <tr>
      <td>78</td>
      <td>bar</td>
      <td>24.034261</td>
    </tr>
    <tr>
      <td>79</td>
      <td>arcade machine</td>
      <td>11.467819</td>
    </tr>
    <tr>
      <td>80</td>
      <td>hovel, hut, hutch, shack, shanty</td>
      <td>10.258017</td>
    </tr>
    <tr>
      <td>81</td>
      <td>bus</td>
      <td>81.375072</td>
    </tr>
    <tr>
      <td>82</td>
      <td>towel</td>
      <td>54.954106</td>
    </tr>
    <tr>
      <td>83</td>
      <td>light</td>
      <td>53.256340</td>
    </tr>
    <tr>
      <td>84</td>
      <td>truck</td>
      <td>29.656645</td>
    </tr>
    <tr>
      <td>85</td>
      <td>tower</td>
      <td>36.864496</td>
    </tr>
    <tr>
      <td>86</td>
      <td>chandelier</td>
      <td>63.787459</td>
    </tr>
    <tr>
      <td>87</td>
      <td>awning, sunshade, sunblind</td>
      <td>23.610311</td>
    </tr>
    <tr>
      <td>88</td>
      <td>street lamp</td>
      <td>29.944617</td>
    </tr>
    <tr>
      <td>89</td>
      <td>booth</td>
      <td>29.360433</td>
    </tr>
    <tr>
      <td>90</td>
      <td>tv</td>
      <td>61.512572</td>
    </tr>
    <tr>
      <td>91</td>
      <td>plane</td>
      <td>53.270513</td>
    </tr>
    <tr>
      <td>92</td>
      <td>dirt track</td>
      <td>4.206758</td>
    </tr>
    <tr>
      <td>93</td>
      <td>clothes</td>
      <td>35.342074</td>
    </tr>
    <tr>
      <td>94</td>
      <td>pole</td>
      <td>20.678348</td>
    </tr>
    <tr>
      <td>95</td>
      <td>land, ground, soil</td>
      <td>3.195710</td>
    </tr>
    <tr>
      <td>96</td>
      <td>bannister, banister, balustrade, balusters, handrail</td>
      <td>17.522631</td>
    </tr>
    <tr>
      <td>97</td>
      <td>escalator, moving staircase, moving stairway</td>
      <td>20.889345</td>
    </tr>
    <tr>
      <td>98</td>
      <td>ottoman, pouf, pouffe, puff, hassock</td>
      <td>47.003450</td>
    </tr>
    <tr>
      <td>99</td>
      <td>bottle</td>
      <td>15.504667</td>
    </tr>
    <tr>
      <td>100</td>
      <td>buffet, counter, sideboard</td>
      <td>26.077572</td>
    </tr>
    <tr>
      <td>101</td>
      <td>poster, posting, placard, notice, bill, card</td>
      <td>30.691103</td>
    </tr>
    <tr>
      <td>102</td>
      <td>stage</td>
      <td>11.744151</td>
    </tr>
    <tr>
      <td>103</td>
      <td>van</td>
      <td>40.161822</td>
    </tr>
    <tr>
      <td>104</td>
      <td>ship</td>
      <td>79.300311</td>
    </tr>
    <tr>
      <td>105</td>
      <td>fountain</td>
      <td>0.112958</td>
    </tr>
    <tr>
      <td>106</td>
      <td>conveyer belt, conveyor belt, conveyer, conveyor, transporter</td>
      <td>60.552373</td>
    </tr>
    <tr>
      <td>107</td>
      <td>canopy</td>
      <td>25.086350</td>
    </tr>
    <tr>
      <td>108</td>
      <td>washer, automatic washer, washing machine</td>
      <td>63.550537</td>
    </tr>
    <tr>
      <td>109</td>
      <td>plaything, toy</td>
      <td>18.290597</td>
    </tr>
    <tr>
      <td>110</td>
      <td>pool</td>
      <td>32.873865</td>
    </tr>
    <tr>
      <td>111</td>
      <td>stool</td>
      <td>39.256308</td>
    </tr>
    <tr>
      <td>112</td>
      <td>barrel, cask</td>
      <td>6.358771</td>
    </tr>
    <tr>
      <td>113</td>
      <td>basket, handbasket</td>
      <td>29.850719</td>
    </tr>
    <tr>
      <td>114</td>
      <td>falls</td>
      <td>57.657161</td>
    </tr>
    <tr>
      <td>115</td>
      <td>tent</td>
      <td>93.717152</td>
    </tr>
    <tr>
      <td>116</td>
      <td>bag</td>
      <td>10.629695</td>
    </tr>
    <tr>
      <td>117</td>
      <td>minibike, motorbike</td>
      <td>56.217901</td>
    </tr>
    <tr>
      <td>118</td>
      <td>cradle</td>
      <td>69.441302</td>
    </tr>
    <tr>
      <td>119</td>
      <td>oven</td>
      <td>38.940583</td>
    </tr>
    <tr>
      <td>120</td>
      <td>ball</td>
      <td>45.543376</td>
    </tr>
    <tr>
      <td>121</td>
      <td>food, solid food</td>
      <td>52.779065</td>
    </tr>
    <tr>
      <td>122</td>
      <td>step, stair</td>
      <td>10.843115</td>
    </tr>
    <tr>
      <td>123</td>
      <td>tank, storage tank</td>
      <td>30.871163</td>
    </tr>
    <tr>
      <td>124</td>
      <td>trade name</td>
      <td>27.908376</td>
    </tr>
    <tr>
      <td>125</td>
      <td>microwave</td>
      <td>32.381977</td>
    </tr>
    <tr>
      <td>126</td>
      <td>pot</td>
      <td>41.040635</td>
    </tr>
    <tr>
      <td>127</td>
      <td>animal</td>
      <td>55.882266</td>
    </tr>
    <tr>
      <td>128</td>
      <td>bicycle</td>
      <td>50.185374</td>
    </tr>
    <tr>
      <td>129</td>
      <td>lake</td>
      <td>0.007605</td>
    </tr>
    <tr>
      <td>130</td>
      <td>dishwasher</td>
      <td>58.970317</td>
    </tr>
    <tr>
      <td>131</td>
      <td>screen</td>
      <td>60.016197</td>
    </tr>
    <tr>
      <td>132</td>
      <td>blanket, cover</td>
      <td>26.963189</td>
    </tr>
    <tr>
      <td>133</td>
      <td>sculpture</td>
      <td>27.667732</td>
    </tr>
    <tr>
      <td>134</td>
      <td>hood, exhaust hood</td>
      <td>58.025458</td>
    </tr>
    <tr>
      <td>135</td>
      <td>sconce</td>
      <td>39.341998</td>
    </tr>
    <tr>
      <td>136</td>
      <td>vase</td>
      <td>31.185747</td>
    </tr>
    <tr>
      <td>137</td>
      <td>traffic light</td>
      <td>23.810429</td>
    </tr>
    <tr>
      <td>138</td>
      <td>tray</td>
      <td>7.244281</td>
    </tr>
    <tr>
      <td>139</td>
      <td>trash can</td>
      <td>30.072544</td>
    </tr>
    <tr>
      <td>140</td>
      <td>fan</td>
      <td>52.113861</td>
    </tr>
    <tr>
      <td>141</td>
      <td>pier</td>
      <td>56.678802</td>
    </tr>
    <tr>
      <td>142</td>
      <td>crt screen</td>
      <td>9.133357</td>
    </tr>
    <tr>
      <td>143</td>
      <td>plate</td>
      <td>38.900407</td>
    </tr>
    <tr>
      <td>144</td>
      <td>monitor</td>
      <td>3.323130</td>
    </tr>
    <tr>
      <td>145</td>
      <td>bulletin board</td>
      <td>52.337659</td>
    </tr>
    <tr>
      <td>146</td>
      <td>shower</td>
      <td>4.692180</td>
    </tr>
    <tr>
      <td>147</td>
      <td>radiator</td>
      <td>43.811464</td>
    </tr>
    <tr>
      <td>148</td>
      <td>glass, drinking glass</td>
      <td>14.036491</td>
    </tr>
    <tr>
      <td>149</td>
      <td>clock</td>
      <td>25.044316</td>
    </tr>
    <tr>
      <td>150</td>
      <td>flag</td>
      <td>40.007933</td>
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

# Get the remote model (fai-m2f-m-ade) from Focoos API
model = focoos.get_remote_model("fai-m2f-m-ade")

# Run inference on an image
predictions = model.infer("./image.jpg", threshold=0.5)

# Output the predictions
print(predictions)
```
