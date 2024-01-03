# Feature_inversion
Feature descriptor inversion for privacy attack during the visual localization

In this repository, we aim on query-side attack rather than the server-side attack.

For the server-side attack, please refer to invSfM_torch repository (https://github.com/ChanhyukYun/invSfM_torch).


In this repository, we provide two types of features; one with Harris corner detector and the other with DoG detector.

For the former case, we only provide weights for SOSNet descriptor, while SIFT, HardNet and SOSNet descriptors are supported with the DoG detector.

The architecture of inversion model is the same for all cases.
