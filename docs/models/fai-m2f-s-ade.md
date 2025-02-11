# fai-m2f-l-ade

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

 - the backbone is [STDC-1](https://github.com/MichaelFan01/STDC-Seg) that shows a trade-off tending to be more efficient.
 - the pixel decoder is a [FPN](https://arxiv.org/abs/1612.03144) getting the features from the stage 2 (1/4 resolution), 3 (1/8 resolution), 4 (1/16 resolution) and 5 (1/32 resolution) of the backbone. Differently from the original paper, for the sake of portability, we removed the deformable attention modules in the pixel decoder, speeding up the inference while only marginally affecting the accuracy.
 - the transformer decoder is a extremely light version of the original, having only 1 decoder layer (instead of 9) and 100 learnable queries.

### Losses
We use the same losses as the original paper:

- loss_ce: Cross-entropy loss for the classification of the classes
- loss_dice: Dice loss for the segmentation of the classes
- loss_mask: A binary cross-entropy loss applied to the predicted segmentation masks

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
      <td>69.973850</td>
    </tr>
    <tr>
      <td>2</td>
      <td>building</td>
      <td>78.431035</td>
    </tr>
    <tr>
      <td>3</td>
      <td>sky</td>
      <td>91.401107</td>
    </tr>
    <tr>
      <td>4</td>
      <td>floor</td>
      <td>73.162280</td>
    </tr>
    <tr>
      <td>5</td>
      <td>tree</td>
      <td>70.535439</td>
    </tr>
    <tr>
      <td>6</td>
      <td>ceiling</td>
      <td>77.258595</td>
    </tr>
    <tr>
      <td>7</td>
      <td>road, route</td>
      <td>78.314172</td>
    </tr>
    <tr>
      <td>8</td>
      <td>bed</td>
      <td>77.755793</td>
    </tr>
    <tr>
      <td>9</td>
      <td>window</td>
      <td>53.012898</td>
    </tr>
    <tr>
      <td>10</td>
      <td>grass</td>
      <td>64.432303</td>
    </tr>
    <tr>
      <td>11</td>
      <td>cabinet</td>
      <td>51.032268</td>
    </tr>
    <tr>
      <td>12</td>
      <td>sidewalk, pavement</td>
      <td>55.642697</td>
    </tr>
    <tr>
      <td>13</td>
      <td>person</td>
      <td>70.461440</td>
    </tr>
    <tr>
      <td>14</td>
      <td>eartd, ground</td>
      <td>30.454824</td>
    </tr>
    <tr>
      <td>15</td>
      <td>door</td>
      <td>36.431782</td>
    </tr>
    <tr>
      <td>16</td>
      <td>table</td>
      <td>43.096636</td>
    </tr>
    <tr>
      <td>17</td>
      <td>mountain, mount</td>
      <td>54.971609</td>
    </tr>
    <tr>
      <td>18</td>
      <td>plant</td>
      <td>45.115711</td>
    </tr>
    <tr>
      <td>19</td>
      <td>curtain</td>
      <td>64.930372</td>
    </tr>
    <tr>
      <td>20</td>
      <td>chair</td>
      <td>40.465565</td>
    </tr>
    <tr>
      <td>21</td>
      <td>car</td>
      <td>76.888113</td>
    </tr>
    <tr>
      <td>22</td>
      <td>water</td>
      <td>41.666148</td>
    </tr>
    <tr>
      <td>23</td>
      <td>painting, picture</td>
      <td>60.099652</td>
    </tr>
    <tr>
      <td>24</td>
      <td>sofa</td>
      <td>49.840449</td>
    </tr>
    <tr>
      <td>25</td>
      <td>shelf</td>
      <td>31.991519</td>
    </tr>
    <tr>
      <td>26</td>
      <td>house</td>
      <td>45.338182</td>
    </tr>
    <tr>
      <td>27</td>
      <td>sea</td>
      <td>51.614250</td>
    </tr>
    <tr>
      <td>28</td>
      <td>mirror</td>
      <td>55.731406</td>
    </tr>
    <tr>
      <td>29</td>
      <td>rug</td>
      <td>51.858072</td>
    </tr>
    <tr>
      <td>30</td>
      <td>field</td>
      <td>23.065903</td>
    </tr>
    <tr>
      <td>31</td>
      <td>armchair</td>
      <td>30.602317</td>
    </tr>
    <tr>
      <td>32</td>
      <td>seat</td>
      <td>50.277596</td>
    </tr>
    <tr>
      <td>33</td>
      <td>fence</td>
      <td>34.439293</td>
    </tr>
    <tr>
      <td>34</td>
      <td>desk</td>
      <td>35.494495</td>
    </tr>
    <tr>
      <td>35</td>
      <td>rock, stone</td>
      <td>39.573617</td>
    </tr>
    <tr>
      <td>36</td>
      <td>wardrobe, closet, press</td>
      <td>51.343586</td>
    </tr>
    <tr>
      <td>37</td>
      <td>lamp</td>
      <td>47.754304</td>
    </tr>
    <tr>
      <td>38</td>
      <td>tub</td>
      <td>71.511291</td>
    </tr>
    <tr>
      <td>39</td>
      <td>rail</td>
      <td>23.280869</td>
    </tr>
    <tr>
      <td>40</td>
      <td>cushion</td>
      <td>39.251768</td>
    </tr>
    <tr>
      <td>41</td>
      <td>base, pedestal, stand</td>
      <td>28.472143</td>
    </tr>
    <tr>
      <td>42</td>
      <td>box</td>
      <td>16.070477</td>
    </tr>
    <tr>
      <td>43</td>
      <td>column, pillar</td>
      <td>37.924454</td>
    </tr>
    <tr>
      <td>44</td>
      <td>signboard, sign</td>
      <td>29.057276</td>
    </tr>
    <tr>
      <td>45</td>
      <td>chest of drawers, chest, bureau, dresser</td>
      <td>36.343963</td>
    </tr>
    <tr>
      <td>46</td>
      <td>counter</td>
      <td>19.595326</td>
    </tr>
    <tr>
      <td>47</td>
      <td>sand</td>
      <td>31.296151</td>
    </tr>
    <tr>
      <td>48</td>
      <td>sink</td>
      <td>54.413180</td>
    </tr>
    <tr>
      <td>49</td>
      <td>skyscraper</td>
      <td>47.583224</td>
    </tr>
    <tr>
      <td>50</td>
      <td>fireplace</td>
      <td>62.204434</td>
    </tr>
    <tr>
      <td>51</td>
      <td>refrigerator, icebox</td>
      <td>54.270643</td>
    </tr>
    <tr>
      <td>52</td>
      <td>grandstand, covered stand</td>
      <td>31.345801</td>
    </tr>
    <tr>
      <td>53</td>
      <td>patd</td>
      <td>22.330369</td>
    </tr>
    <tr>
      <td>54</td>
      <td>stairs</td>
      <td>20.323718</td>
    </tr>
    <tr>
      <td>55</td>
      <td>runway</td>
      <td>63.892811</td>
    </tr>
    <tr>
      <td>56</td>
      <td>case, display case, showcase, vitrine</td>
      <td>34.649422</td>
    </tr>
    <tr>
      <td>57</td>
      <td>pool table, billiard table, snooker table</td>
      <td>85.365581</td>
    </tr>
    <tr>
      <td>58</td>
      <td>pillow</td>
      <td>46.426184</td>
    </tr>
    <tr>
      <td>59</td>
      <td>screen door, screen</td>
      <td>57.292321</td>
    </tr>
    <tr>
      <td>60</td>
      <td>stairway, staircase</td>
      <td>28.904954</td>
    </tr>
    <tr>
      <td>61</td>
      <td>river</td>
      <td>16.681450</td>
    </tr>
    <tr>
      <td>62</td>
      <td>bridge, span</td>
      <td>52.791513</td>
    </tr>
    <tr>
      <td>63</td>
      <td>bookcase</td>
      <td>26.722881</td>
    </tr>
    <tr>
      <td>64</td>
      <td>blind, screen</td>
      <td>36.787453</td>
    </tr>
    <tr>
      <td>65</td>
      <td>coffee table</td>
      <td>41.603442</td>
    </tr>
    <tr>
      <td>66</td>
      <td>toilet, can, commode, crapper, pot, potty, stool, tdrone</td>
      <td>75.753455</td>
    </tr>
    <tr>
      <td>67</td>
      <td>flower</td>
      <td>30.200230</td>
    </tr>
    <tr>
      <td>68</td>
      <td>book</td>
      <td>37.602484</td>
    </tr>
    <tr>
      <td>69</td>
      <td>hill</td>
      <td>5.509057</td>
    </tr>
    <tr>
      <td>70</td>
      <td>bench</td>
      <td>29.331054</td>
    </tr>
    <tr>
      <td>71</td>
      <td>countertop</td>
      <td>46.661677</td>
    </tr>
    <tr>
      <td>72</td>
      <td>stove</td>
      <td>58.972851</td>
    </tr>
    <tr>
      <td>73</td>
      <td>palm, palm tree</td>
      <td>48.317300</td>
    </tr>
    <tr>
      <td>74</td>
      <td>kitchen island</td>
      <td>25.279206</td>
    </tr>
    <tr>
      <td>75</td>
      <td>computer</td>
      <td>49.335666</td>
    </tr>
    <tr>
      <td>76</td>
      <td>swivel chair</td>
      <td>34.845392</td>
    </tr>
    <tr>
      <td>77</td>
      <td>boat</td>
      <td>48.521646</td>
    </tr>
    <tr>
      <td>78</td>
      <td>bar</td>
      <td>30.174155</td>
    </tr>
    <tr>
      <td>79</td>
      <td>arcade machine</td>
      <td>24.721694</td>
    </tr>
    <tr>
      <td>80</td>
      <td>hovel, hut, hutch, shack, shanty</td>
      <td>32.843717</td>
    </tr>
    <tr>
      <td>81</td>
      <td>bus</td>
      <td>82.174778</td>
    </tr>
    <tr>
      <td>82</td>
      <td>towel</td>
      <td>46.050430</td>
    </tr>
    <tr>
      <td>83</td>
      <td>light</td>
      <td>30.983118</td>
    </tr>
    <tr>
      <td>84</td>
      <td>truck</td>
      <td>23.456256</td>
    </tr>
    <tr>
      <td>85</td>
      <td>tower</td>
      <td>32.147803</td>
    </tr>
    <tr>
      <td>86</td>
      <td>chandelier</td>
      <td>54.045160</td>
    </tr>
    <tr>
      <td>87</td>
      <td>awning, sunshade, sunblind</td>
      <td>18.526182</td>
    </tr>
    <tr>
      <td>88</td>
      <td>street lamp</td>
      <td>13.641714</td>
    </tr>
    <tr>
      <td>89</td>
      <td>bootd</td>
      <td>60.471570</td>
    </tr>
    <tr>
      <td>90</td>
      <td>tv</td>
      <td>55.530715</td>
    </tr>
    <tr>
      <td>91</td>
      <td>plane</td>
      <td>42.894525</td>
    </tr>
    <tr>
      <td>92</td>
      <td>dirt track</td>
      <td>0.001787</td>
    </tr>
    <tr>
      <td>93</td>
      <td>clotdes</td>
      <td>30.124455</td>
    </tr>
    <tr>
      <td>94</td>
      <td>pole</td>
      <td>11.280532</td>
    </tr>
    <tr>
      <td>95</td>
      <td>land, ground, soil</td>
      <td>4.243296</td>
    </tr>
    <tr>
      <td>96</td>
      <td>bannister, banister, balustrade, balusters, handrail</td>
      <td>9.922319</td>
    </tr>
    <tr>
      <td>97</td>
      <td>escalator, moving staircase, moving stairway</td>
      <td>19.186240</td>
    </tr>
    <tr>
      <td>98</td>
      <td>ottoman, pouf, pouffe, puff, hassock</td>
      <td>30.352586</td>
    </tr>
    <tr>
      <td>99</td>
      <td>bottle</td>
      <td>11.872842</td>
    </tr>
    <tr>
      <td>100</td>
      <td>buffet, counter, sideboard</td>
      <td>34.547476</td>
    </tr>
    <tr>
      <td>101</td>
      <td>poster, posting, placard, notice, bill, card</td>
      <td>15.081001</td>
    </tr>
    <tr>
      <td>102</td>
      <td>stage</td>
      <td>17.466091</td>
    </tr>
    <tr>
      <td>103</td>
      <td>van</td>
      <td>39.027877</td>
    </tr>
    <tr>
      <td>104</td>
      <td>ship</td>
      <td>66.778301</td>
    </tr>
    <tr>
      <td>105</td>
      <td>fountain</td>
      <td>18.879113</td>
    </tr>
    <tr>
      <td>106</td>
      <td>conveyer belt, conveyor belt, conveyer, conveyor, transporter</td>
      <td>67.580228</td>
    </tr>
    <tr>
      <td>107</td>
      <td>canopy</td>
      <td>25.654567</td>
    </tr>
    <tr>
      <td>108</td>
      <td>washer, automatic washer, washing machine</td>
      <td>60.187881</td>
    </tr>
    <tr>
      <td>109</td>
      <td>playtding, toy</td>
      <td>13.836259</td>
    </tr>
    <tr>
      <td>110</td>
      <td>pool</td>
      <td>28.796494</td>
    </tr>
    <tr>
      <td>111</td>
      <td>stool</td>
      <td>26.432746</td>
    </tr>
    <tr>
      <td>112</td>
      <td>barrel, cask</td>
      <td>43.777156</td>
    </tr>
    <tr>
      <td>113</td>
      <td>basket, handbasket</td>
      <td>19.144369</td>
    </tr>
    <tr>
      <td>114</td>
      <td>falls</td>
      <td>47.131198</td>
    </tr>
    <tr>
      <td>115</td>
      <td>tent</td>
      <td>88.431441</td>
    </tr>
    <tr>
      <td>116</td>
      <td>bag</td>
      <td>7.634387</td>
    </tr>
    <tr>
      <td>117</td>
      <td>minibike, motorbike</td>
      <td>40.625528</td>
    </tr>
    <tr>
      <td>118</td>
      <td>cradle</td>
      <td>54.247514</td>
    </tr>
    <tr>
      <td>119</td>
      <td>oven</td>
      <td>33.695444</td>
    </tr>
    <tr>
      <td>120</td>
      <td>ball</td>
      <td>36.066130</td>
    </tr>
    <tr>
      <td>121</td>
      <td>food, solid food</td>
      <td>50.837348</td>
    </tr>
    <tr>
      <td>122</td>
      <td>step, stair</td>
      <td>13.071184</td>
    </tr>
    <tr>
      <td>123</td>
      <td>tank, storage tank</td>
      <td>43.042742</td>
    </tr>
    <tr>
      <td>124</td>
      <td>trade name</td>
      <td>21.579095</td>
    </tr>
    <tr>
      <td>125</td>
      <td>microwave</td>
      <td>32.179626</td>
    </tr>
    <tr>
      <td>126</td>
      <td>pot</td>
      <td>27.438416</td>
    </tr>
    <tr>
      <td>127</td>
      <td>animal</td>
      <td>55.993825</td>
    </tr>
    <tr>
      <td>128</td>
      <td>bicycle</td>
      <td>38.273475</td>
    </tr>
    <tr>
      <td>129</td>
      <td>lake</td>
      <td>35.704904</td>
    </tr>
    <tr>
      <td>130</td>
      <td>dishwasher</td>
      <td>37.616793</td>
    </tr>
    <tr>
      <td>131</td>
      <td>screen</td>
      <td>57.100955</td>
    </tr>
    <tr>
      <td>132</td>
      <td>blanket, cover</td>
      <td>15.560568</td>
    </tr>
    <tr>
      <td>133</td>
      <td>sculpture</td>
      <td>31.317035</td>
    </tr>
    <tr>
      <td>134</td>
      <td>hood, exhaust hood</td>
      <td>49.290385</td>
    </tr>
    <tr>
      <td>135</td>
      <td>sconce</td>
      <td>29.971644</td>
    </tr>
    <tr>
      <td>136</td>
      <td>vase</td>
      <td>24.983318</td>
    </tr>
    <tr>
      <td>137</td>
      <td>traffic light</td>
      <td>17.806663</td>
    </tr>
    <tr>
      <td>138</td>
      <td>tray</td>
      <td>5.720345</td>
    </tr>
    <tr>
      <td>139</td>
      <td>trash can</td>
      <td>28.621136</td>
    </tr>
    <tr>
      <td>140</td>
      <td>fan</td>
      <td>39.083851</td>
    </tr>
    <tr>
      <td>141</td>
      <td>pier</td>
      <td>51.310956</td>
    </tr>
    <tr>
      <td>142</td>
      <td>crt screen</td>
      <td>0.858346</td>
    </tr>
    <tr>
      <td>143</td>
      <td>plate</td>
      <td>35.344330</td>
    </tr>
    <tr>
      <td>144</td>
      <td>monitor</td>
      <td>1.994270</td>
    </tr>
    <tr>
      <td>145</td>
      <td>bulletin board</td>
      <td>35.468027</td>
    </tr>
    <tr>
      <td>146</td>
      <td>shower</td>
      <td>1.090403</td>
    </tr>
    <tr>
      <td>147</td>
      <td>radiator</td>
      <td>42.574652</td>
    </tr>
    <tr>
      <td>148</td>
      <td>glass, drinking glass</td>
      <td>8.510381</td>
    </tr>
    <tr>
      <td>149</td>
      <td>clock</td>
      <td>14.128872</td>
    </tr>
    <tr>
      <td>150</td>
      <td>flag</td>
      <td>24.098100</td>
    </tr>
  </tbody>
</table>
</table>

</div>


## What are you waiting? Try it!
```python
from focoos import Focoos
import os

# Initialize the Focoos client with your API key
focoos = Focoos(api_key=os.getenv("FOCOOS_API_KEY"))

# Get the remote model (fai-m2f-s-ade) from Focoos API
model = focoos.get_remote_model("fai-m2f-s-ade")

# Run inference on an image
predictions = model.infer("./image.jpg", threshold=0.5)

# Output the predictions
print(predictions)
```
