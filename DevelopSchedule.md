# Version 0.2.0 Develop Schedule

## Abstract 

- Author: Yifei Xu 
- Date: 09/25/2021 - 09/30/2021

## Files

- Modified
    - train.py
- Add
    - 

### models/networks.py 



##### Change accordingly 

- Replace network name
    - Replace `DCGenerator` to `GeneratorConv2D`
    - Replace `DCEncoder` to `EncoderConv2D`
    - Replace `MNISTGenerator` to `GeneratorMnist`
    - Replace `DCEBM` to `EnergyBasedNetworkConv2D`
- Add registry 
    - `model/builder.py`: add `NETWORK` and `SAMPLINGS`, `get_encoder` and `get_sampling`.
- Change datasets 
    - 