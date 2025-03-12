# fai-bf-m-ade

## Overview
The models is a [BiSeNetFormer](https://arxiv.org/abs/2404.09570)  model otimized by [FocoosAI](https://focoos.ai) for the [ADE20K dataset](https://groups.csail.mit.edu/vision/datasets/ADE20K/). It is a semantic segmentation model able to segment 150 classes, comprising both stuff (sky, road, etc.) and thing (dog, cat, car, etc.).


## Benchmark
![Benchmark Comparison](./fai-ade.png)
Note: FPS are computed on NVIDIA T4 using TensorRT and image size 640x640.

## Model Details
The model is based on the [BiSeNetFormer](https://arxiv.org/abs/2404.09570) architecture. It is a segmentation model that uses a two-stream feature extractor and a transformer-based decoder architecture.
Similarly to [Mask2Former](https://arxiv.org/abs/2112.01527), BiSeNetFormer uses a mask-classification approach, where the prediction is made by a set of segmentation mask with associated class probabilities. 

### Neural Network Architecture
The [BiSeNetFormer](https://arxiv.org/abs/2404.09570) FocoosAI implementation is the original neural network architecture, fully described in this [paper](https://arxiv.org/abs/2404.09570).

BiSeNetFormer has the three main components: a *spatial path* for extracting high-resolution features, a *context path* for capturing semantic visual features, and a *transformer decoder* for generating segment embeddings. 

![alt text](./bisenetformer.png)

In this implementation:

- the backbone is [STDC-2](https://github.com/MichaelFan01/STDC-Seg) that show an amazing trade-off between performance and efficiency.
- the transformer decoder has 4 decoder layers and 100 learnable queries.

### Losses
We use the same losses as the original paper:

- loss_ce: Cross-entropy loss for the classification of the classes
- loss_dice: Dice loss for the segmentation of the classes
- loss_mask: A binary cross-entropy loss applied to the predicted segmentation masks

These losses are applied to each output of the transformer decoder, meaning that we apply it on the output and on each auxiliary output of the transformer decoder layers.
Please refer to the [BiSeNetFormer](https://arxiv.org/abs/2404.09570) for more details.

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
      <th>Segmentation AP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>wall</td>
      <td>73.766386</td>
    </tr>
    <tr>
      <td>2</td>
      <td>building</td>
      <td>78.494310</td>
    </tr>
    <tr>
      <td>3</td>
      <td>sky</td>
      <td>93.410361</td>
    </tr>
    <tr>
      <td>4</td>
      <td>floor</td>
      <td>79.551193</td>
    </tr>
    <tr>
      <td>5</td>
      <td>tree</td>
      <td>72.505667</td>
    </tr>
    <tr>
      <td>6</td>
      <td>ceiling</td>
      <td>81.497542</td>
    </tr>
    <tr>
      <td>7</td>
      <td>road, route</td>
      <td>79.954237</td>
    </tr>
    <tr>
      <td>8</td>
      <td>bed</td>
      <td>86.884124</td>
    </tr>
    <tr>
      <td>9</td>
      <td>window </td>
      <td>57.782973</td>
    </tr>
    <tr>
      <td>10</td>
      <td>grass</td>
      <td>68.321229</td>
    </tr>
    <tr>
      <td>11</td>
      <td>cabinet</td>
      <td>57.602831</td>
    </tr>
    <tr>
      <td>12</td>
      <td>sidewalk, pavement</td>
      <td>62.884492</td>
    </tr>
    <tr>
      <td>13</td>
      <td>person</td>
      <td>75.333445</td>
    </tr>
    <tr>
      <td>14</td>
      <td>earth, ground</td>
      <td>32.365395</td>
    </tr>
    <tr>
      <td>15</td>
      <td>door</td>
      <td>42.766662</td>
    </tr>
    <tr>
      <td>16</td>
      <td>table</td>
      <td>54.505274</td>
    </tr>
    <tr>
      <td>17</td>
      <td>mountain, mount</td>
      <td>56.006902</td>
    </tr>
    <tr>
      <td>18</td>
      <td>plant</td>
      <td>52.419150</td>
    </tr>
    <tr>
      <td>19</td>
      <td>curtain</td>
      <td>67.454516</td>
    </tr>
    <tr>
      <td>20</td>
      <td>chair</td>
      <td>48.881459</td>
    </tr>
    <tr>
      <td>21</td>
      <td>car</td>
      <td>80.253139</td>
    </tr>
    <tr>
      <td>22</td>
      <td>water</td>
      <td>53.351696</td>
    </tr>
    <tr>
      <td>23</td>
      <td>painting, picture</td>
      <td>59.611184</td>
    </tr>
    <tr>
      <td>24</td>
      <td>sofa</td>
      <td>56.758410</td>
    </tr>
    <tr>
      <td>25</td>
      <td>shelf</td>
      <td>33.939530</td>
    </tr>
    <tr>
      <td>26</td>
      <td>house</td>
      <td>47.170888</td>
    </tr>
    <tr>
      <td>27</td>
      <td>sea</td>
      <td>57.485670</td>
    </tr>
    <tr>
      <td>28</td>
      <td>mirror</td>
      <td>56.591294</td>
    </tr>
    <tr>
      <td>29</td>
      <td>rug</td>
      <td>59.663102</td>
    </tr>
    <tr>
      <td>30</td>
      <td>field</td>
      <td>39.517764</td>
    </tr>
    <tr>
      <td>31</td>
      <td>armchair</td>
      <td>37.758644</td>
    </tr>
    <tr>
      <td>32</td>
      <td>seat</td>
      <td>53.283106</td>
    </tr>
    <tr>
      <td>33</td>
      <td>fence</td>
      <td>42.843060</td>
    </tr>
    <tr>
      <td>34</td>
      <td>desk</td>
      <td>43.395206</td>
    </tr>
    <tr>
      <td>35</td>
      <td>rock, stone</td>
      <td>40.691055</td>
    </tr>
    <tr>
      <td>36</td>
      <td>wardrobe, closet, press</td>
      <td>44.121024</td>
    </tr>
    <tr>
      <td>37</td>
      <td>lamp</td>
      <td>58.026347</td>
    </tr>
    <tr>
      <td>38</td>
      <td>tub</td>
      <td>79.702090</td>
    </tr>
    <tr>
      <td>39</td>
      <td>rail</td>
      <td>30.872559</td>
    </tr>
    <tr>
      <td>40</td>
      <td>cushion</td>
      <td>48.281128</td>
    </tr>
    <tr>
      <td>41</td>
      <td>base, pedestal, stand</td>
      <td>26.747704</td>
    </tr>
    <tr>
      <td>42</td>
      <td>box</td>
      <td>15.263754</td>
    </tr>
    <tr>
      <td>43</td>
      <td>column, pillar</td>
      <td>30.738794</td>
    </tr>
    <tr>
      <td>44</td>
      <td>signboard, sign</td>
      <td>32.488200</td>
    </tr>
    <tr>
      <td>45</td>
      <td>chest of drawers, chest, bureau, dresser</td>
      <td>41.765120</td>
    </tr>
    <tr>
      <td>46</td>
      <td>counter</td>
      <td>23.951655</td>
    </tr>
    <tr>
      <td>47</td>
      <td>sand</td>
      <td>30.733368</td>
    </tr>
    <tr>
      <td>48</td>
      <td>sink</td>
      <td>66.012565</td>
    </tr>
    <tr>
      <td>49</td>
      <td>skyscraper</td>
      <td>43.017489</td>
    </tr>
    <tr>
      <td>50</td>
      <td>fireplace</td>
      <td>68.527845</td>
    </tr>
    <tr>
      <td>51</td>
      <td>refrigerator, icebox</td>
      <td>64.140162</td>
    </tr>
    <tr>
      <td>52</td>
      <td>grandstand, covered stand</td>
      <td>42.008015</td>
    </tr>
    <tr>
      <td>53</td>
      <td>path</td>
      <td>20.711066</td>
    </tr>
    <tr>
      <td>54</td>
      <td>stairs</td>
      <td>19.888671</td>
    </tr>
    <tr>
      <td>55</td>
      <td>runway</td>
      <td>72.573838</td>
    </tr>
    <tr>
      <td>56</td>
      <td>case, display case, showcase, vitrine</td>
      <td>46.085263</td>
    </tr>
    <tr>
      <td>57</td>
      <td>pool table, billiard table, snooker table</td>
      <td>87.221437</td>
    </tr>
    <tr>
      <td>58</td>
      <td>pillow</td>
      <td>50.546325</td>
    </tr>
    <tr>
      <td>59</td>
      <td>screen door, screen</td>
      <td>62.427008</td>
    </tr>
    <tr>
      <td>60</td>
      <td>stairway, staircase</td>
      <td>24.685976</td>
    </tr>
    <tr>
      <td>61</td>
      <td>river</td>
      <td>19.006379</td>
    </tr>
    <tr>
      <td>62</td>
      <td>bridge, span</td>
      <td>26.688646</td>
    </tr>
    <tr>
      <td>63</td>
      <td>bookcase</td>
      <td>31.981159</td>
    </tr>
    <tr>
      <td>64</td>
      <td>blind, screen</td>
      <td>37.324359</td>
    </tr>
    <tr>
      <td>65</td>
      <td>coffee table</td>
      <td>61.400744</td>
    </tr>
    <tr>
      <td>66</td>
      <td>toilet, can, commode, crapper, pot, potty, stool, throne</td>
      <td>82.339204</td>
    </tr>
    <tr>
      <td>67</td>
      <td>flower</td>
      <td>36.046776</td>
    </tr>
    <tr>
      <td>68</td>
      <td>book</td>
      <td>44.102006</td>
    </tr>
    <tr>
      <td>69</td>
      <td>hill</td>
      <td>9.490701</td>
    </tr>
    <tr>
      <td>70</td>
      <td>bench</td>
      <td>35.926753</td>
    </tr>
    <tr>
      <td>71</td>
      <td>countertop</td>
      <td>47.525603</td>
    </tr>
    <tr>
      <td>72</td>
      <td>stove</td>
      <td>62.960413</td>
    </tr>
    <tr>
      <td>73</td>
      <td>palm, palm tree</td>
      <td>44.101300</td>
    </tr>
    <tr>
      <td>74</td>
      <td>kitchen island</td>
      <td>39.852260</td>
    </tr>
    <tr>
      <td>75</td>
      <td>computer</td>
      <td>56.288398</td>
    </tr>
    <tr>
      <td>76</td>
      <td>swivel chair</td>
      <td>32.996698</td>
    </tr>
    <tr>
      <td>77</td>
      <td>boat</td>
      <td>30.770802</td>
    </tr>
    <tr>
      <td>78</td>
      <td>bar</td>
      <td>21.874256</td>
    </tr>
    <tr>
      <td>79</td>
      <td>arcade machine</td>
      <td>62.693573</td>
    </tr>
    <tr>
      <td>80</td>
      <td>hovel, hut, hutch, shack, shanty</td>
      <td>29.336995</td>
    </tr>
    <tr>
      <td>81</td>
      <td>bus</td>
      <td>76.876302</td>
    </tr>
    <tr>
      <td>82</td>
      <td>towel</td>
      <td>42.404806</td>
    </tr>
    <tr>
      <td>83</td>
      <td>light</td>
      <td>51.937168</td>
    </tr>
    <tr>
      <td>84</td>
      <td>truck</td>
      <td>20.424432</td>
    </tr>
    <tr>
      <td>85</td>
      <td>tower</td>
      <td>5.532333</td>
    </tr>
    <tr>
      <td>86</td>
      <td>chandelier</td>
      <td>66.059028</td>
    </tr>
    <tr>
      <td>87</td>
      <td>awning, sunshade, sunblind</td>
      <td>23.081661</td>
    </tr>
    <tr>
      <td>88</td>
      <td>street lamp</td>
      <td>26.329727</td>
    </tr>
    <tr>
      <td>89</td>
      <td>booth</td>
      <td>36.267256</td>
    </tr>
    <tr>
      <td>90</td>
      <td>tv</td>
      <td>62.627177</td>
    </tr>
    <tr>
      <td>91</td>
      <td>plane</td>
      <td>50.491534</td>
    </tr>
    <tr>
      <td>92</td>
      <td>dirt track</td>
      <td>6.332212</td>
    </tr>
    <tr>
      <td>93</td>
      <td>clothes</td>
      <td>25.463751</td>
    </tr>
    <tr>
      <td>94</td>
      <td>pole</td>
      <td>23.596586</td>
    </tr>
    <tr>
      <td>95</td>
      <td>land, ground, soil</td>
      <td>5.693100</td>
    </tr>
    <tr>
      <td>96</td>
      <td>bannister, banister, balustrade, balusters, handrail</td>
      <td>7.794173</td>
    </tr>
    <tr>
      <td>97</td>
      <td>escalator, moving staircase, moving stairway</td>
      <td>55.479096</td>
    </tr>
    <tr>
      <td>98</td>
      <td>ottoman, pouf, pouffe, puff, hassock</td>
      <td>39.890859</td>
    </tr>
    <tr>
      <td>99</td>
      <td>bottle</td>
      <td>17.481542</td>
    </tr>
    <tr>
      <td>100</td>
      <td>buffet, counter, sideboard</td>
      <td>40.663620</td>
    </tr>
    <tr>
      <td>101</td>
      <td>poster, posting, placard, notice, bill, card</td>
      <td>23.602198</td>
    </tr>
    <tr>
      <td>102</td>
      <td>stage</td>
      <td>7.491402</td>
    </tr>
    <tr>
      <td>103</td>
      <td>van</td>
      <td>37.687056</td>
    </tr>
    <tr>
      <td>104</td>
      <td>ship</td>
      <td>56.925203</td>
    </tr>
    <tr>
      <td>105</td>
      <td>fountain</td>
      <td>19.377751</td>
    </tr>
    <tr>
      <td>106</td>
      <td>conveyer belt, conveyor belt, conveyer, conveyor, transporter</td>
      <td>54.896354</td>
    </tr>
    <tr>
      <td>107</td>
      <td>canopy</td>
      <td>30.170667</td>
    </tr>
    <tr>
      <td>108</td>
      <td>washer, automatic washer, washing machine</td>
      <td>61.346716</td>
    </tr>
    <tr>
      <td>109</td>
      <td>plaything, toy</td>
      <td>19.173092</td>
    </tr>
    <tr>
      <td>110</td>
      <td>pool</td>
      <td>46.819571</td>
    </tr>
    <tr>
      <td>111</td>
      <td>stool</td>
      <td>35.376157</td>
    </tr>
    <tr>
      <td>112</td>
      <td>barrel, cask</td>
      <td>4.432488</td>
    </tr>
    <tr>
      <td>113</td>
      <td>basket, handbasket</td>
      <td>20.526495</td>
    </tr>
    <tr>
      <td>114</td>
      <td>falls</td>
      <td>55.469403</td>
    </tr>
    <tr>
      <td>115</td>
      <td>tent</td>
      <td>49.066246</td>
    </tr>
    <tr>
      <td>116</td>
      <td>bag</td>
      <td>11.269300</td>
    </tr>
    <tr>
      <td>117</td>
      <td>minibike, motorbike</td>
      <td>65.135034</td>
    </tr>
    <tr>
      <td>118</td>
      <td>cradle</td>
      <td>79.832799</td>
    </tr>
    <tr>
      <td>119</td>
      <td>oven</td>
      <td>40.335734</td>
    </tr>
    <tr>
      <td>120</td>
      <td>ball</td>
      <td>39.559197</td>
    </tr>
    <tr>
      <td>121</td>
      <td>food, solid food</td>
      <td>53.097366</td>
    </tr>
    <tr>
      <td>122</td>
      <td>step, stair</td>
      <td>6.321819</td>
    </tr>
    <tr>
      <td>123</td>
      <td>tank, storage tank</td>
      <td>28.717885</td>
    </tr>
    <tr>
      <td>124</td>
      <td>trade name</td>
      <td>28.509980</td>
    </tr>
    <tr>
      <td>125</td>
      <td>microwave</td>
      <td>40.076241</td>
    </tr>
    <tr>
      <td>126</td>
      <td>pot</td>
      <td>43.471795</td>
    </tr>
    <tr>
      <td>127</td>
      <td>animal</td>
      <td>55.538679</td>
    </tr>
    <tr>
      <td>128</td>
      <td>bicycle</td>
      <td>50.574975</td>
    </tr>
    <tr>
      <td>129</td>
      <td>lake</td>
      <td>62.454958</td>
    </tr>
    <tr>
      <td>130</td>
      <td>dishwasher</td>
      <td>48.958419</td>
    </tr>
    <tr>
      <td>131</td>
      <td>screen</td>
      <td>78.679387</td>
    </tr>
    <tr>
      <td>132</td>
      <td>blanket, cover</td>
      <td>16.172520</td>
    </tr>
    <tr>
      <td>133</td>
      <td>sculpture</td>
      <td>31.841992</td>
    </tr>
    <tr>
      <td>134</td>
      <td>hood, exhaust hood</td>
      <td>64.563576</td>
    </tr>
    <tr>
      <td>135</td>
      <td>sconce</td>
      <td>37.741058</td>
    </tr>
    <tr>
      <td>136</td>
      <td>vase</td>
      <td>29.226198</td>
    </tr>
    <tr>
      <td>137</td>
      <td>traffic light</td>
      <td>22.322043</td>
    </tr>
    <tr>
      <td>138</td>
      <td>tray</td>
      <td>8.160090</td>
    </tr>
    <tr>
      <td>139</td>
      <td>trash can</td>
      <td>30.932332</td>
    </tr>
    <tr>
      <td>140</td>
      <td>fan</td>
      <td>54.497612</td>
    </tr>
    <tr>
      <td>141</td>
      <td>pier</td>
      <td>29.278075</td>
    </tr>
    <tr>
      <td>142</td>
      <td>crt screen</td>
      <td>0.370523</td>
    </tr>
    <tr>
      <td>143</td>
      <td>plate</td>
      <td>33.845567</td>
    </tr>
    <tr>
      <td>144</td>
      <td>monitor</td>
      <td>3.128528</td>
    </tr>
    <tr>
      <td>145</td>
      <td>bulletin board</td>
      <td>48.651906</td>
    </tr>
    <tr>
      <td>146</td>
      <td>shower</td>
      <td>8.242905</td>
    </tr>
    <tr>
      <td>147</td>
      <td>radiator</td>
      <td>57.948424</td>
    </tr>
    <tr>
      <td>148</td>
      <td>glass, drinking glass</td>
      <td>12.038441</td>
    </tr>
    <tr>
      <td>149</td>
      <td>clock</td>
      <td>18.156148</td>
    </tr>
    <tr>
      <td>150</td>
      <td>flag</td>
      <td>28.601342</td>
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

# Get the remote model (fai-bf-m-ade) from Focoos API
model = focoos.get_remote_model("fai-bf-m-ade")

# Run inference on an image
predictions = model.infer("./image.jpg", threshold=0.5)

# Output the predictions
print(predictions)
```
