# Failure Mode Analysis & Engineering Mitigations

Detailed inspection of 5 representative failure modes encountered by the Helmet Safety Violation Detector on held-out test data.

### Example 1: Small Scale & Distant Workers
![Failure Case 1](outputs/failures\failure_case_1.jpg)

- **Manifestation**: Worker head occupies less than 20x20 pixels in the scene.
- **Root Cause Analysis**: Receptive field downsampling in convolutional feature pyramids (P3-P5) aggregates background context with tiny head features, diluting helmet textural cues.
- **Engineering Mitigation**: Incorporate Sliced Aided Hyper Inference (SAHI) during inference, train with higher input resolution (e.g. 1280x1280), or utilize a P2 detection head for micro-objects.

### Example 2: Severe Foreground Occlusion
![Failure Case 2](outputs/failures\failure_case_2.jpg)

- **Manifestation**: Worker is partially obscured by scaffolding, metal pipes, or safety nets.
- **Root Cause Analysis**: Bounding box IoU drops below detection threshold due to fragmented visual boundaries; feature extractor fails to aggregate sufficient unobstructed helmet area.
- **Engineering Mitigation**: Apply synthetic CutMix and Random Erasing augmentations during training to force the model to classify using partial visible contours.

### Example 3: Headwear Ambiguity (Caps / Hoodies)
![Failure Case 3](outputs/failures\failure_case_3.jpg)

- **Manifestation**: Worker wearing a baseball cap or hood falsely classified as a helmet.
- **Root Cause Analysis**: Curved brim and dome structure of a cap resemble hard-hat geometry in lower lighting or low-contrast conditions.
- **Engineering Mitigation**: Hard Negative Mining: Collect non-compliant workers with caps, turbans, and beanies in the training set and label them strictly as 'no_helmet'.

### Example 4: High Contrast & Harsh Shadowing
![Failure Case 4](outputs/failures\failure_case_4.jpg)

- **Manifestation**: Extreme sunlight casting pitch-black shadows over the head beneath crane or ceiling structures.
- **Root Cause Analysis**: Clipping in dynamic range causes loss of color saturation and texture in shadowed regions, making reflective strips and helmet curvature invisible.
- **Engineering Mitigation**: Introduce CLAHE (Contrast Limited Adaptive Histogram Equalization) preprocessing and HSV/brightness jittering during training.

### Example 5: Motion Blur & Rapid Camera Panning
![Failure Case 5](outputs/failures\failure_case_5.jpg)

- **Manifestation**: Fast worker movement or camera vibration causes edge smearing.
- **Root Cause Analysis**: Gradient smear across pixel boundaries attenuates high-frequency hard-hat edge features.
- **Engineering Mitigation**: Train with synthetic directional motion blur augmentations (Albumentations) and apply temporal multi-frame aggregation across consecutive frames.
