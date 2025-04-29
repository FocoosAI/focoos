## List of things to do

- [-] Implement the model wrapper that will offer high-level API (such as train, predict, export - as in figma)
  - [x] Train/Predict/Test
  - [ ] Missing export function

- [ ] Introduce additional models from anyma
  - [x] RTDETR
  - [ ] Maskformer
  - [ ] PEMFormer
  - [ ] BisenetFormer

- [X] Clean the backbones to support all of them
  - [X] Unify resnet and presnet weights and models
  - [X] Clean and refactor the code (in backbones)
  - [X] Clean and refactor the code (in nn)


- [ ] Implement the hub with focoos
    - [ ] Download config from the hub
    - [ ] Store pretrained weights on the hub
    - [ ] Store the final metadata on the hub as we did for focoos and remove wandb

- [ ] Add the proper copyrights (especially from detectron2)
