# An interpretable DCN-Transformer hybrid model with attribution for mineral prospectivity mapping
This study develops an interpretable attribution-guided DCN–Transformer model for mineral prospectivity mapping. The model integrates a deformable convolutional network to capture spatially variable local patterns, including irregular geochemical anomalies and ore-controlling structures, with a Transformer encoder to model broader spatial dependencies. A trainable factor-attribution branch applies DeepLIFT relative to a non-mineralized reference state to quantify the signed contribution of each predictive factor. These contributions are further mapped into a spatial attribution map, which modulates the Key representation and serves as an attention bias, while the Query and Value representations are derived from DCN features. The fused features are subsequently processed by the Transformer to estimate mineralization probability. This design enables joint learning of local structural characteristics, spatial interactions, and factor-level effects. It also supports interpretation at multiple levels through global and deposit-specific DeepLIFT contributions, Grad-CAM response maps, and Transformer attention patterns.

## Environment
This code was developed and tested in the following environment:
**Python**: 3.9.13  
**PyTorch**: 2.8.0 (CUDA 11.8/12.1)  

## Requirements
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.23.0
scikit-learn>=1.2.0
joblib>=1.2.0
matplotlib>=3.6.0
pandas>=1.5.0
tqdm>=4.64.0
Pillow>=9.4.0
einops>=0.6.0

## File Structure & Functions
```
research/DCN-attribution-Transformer/
├── build_samples.py                   #  This script prepares training and validation samples for the mineral prospectivity study
├── spatial_grouped_fivefold_cv.py                   #  This script implements spatial 5-fold cross-validation
├── CNN.py                   #  This script runs the CNN ablation experiment for the study
├── CNN_Transformer.py                   #  This script runs the CNN-Transformer ablation experiment for the study
├── Attribution_Guided_CNN_Transformer.py                   #  This script runs the Attribution-Guided CNN-Transformer ablation experiment for the study
├── DCN.py                   #  This script runs the DCN-only ablation experiment for the study
├── Transformer.py                   #  This script runs the Transformer-only ablation experiment for the study
├── DCN_Transformer.py                #  This script runs the DCN-Transformer ablation experiment for the study
├── attribution_guided_model.py                #  This module defines the attribution-guided DCN-Transformer model used in the study
├── dcnv2_model.py               #  This module defines the DCNv2-based feature extraction components used in the study
├── Dense_transformers.py               #  This module defines the dense transformer components used in the study
├── Attribution_Guided_DCN_Transformer.py            #  This script trains and evaluates the Attribution-Guided model for the study
├── visualize_transformer_attention.py            #  This script visualizes attention maps produced by the Transformer model
├── visualize_deeplift_global.py            #  This script visualizes DeepLIFT feature attributions with beeswarm plots
├── visualize_gradcam_deposits.py            #  This script visualizes Grad-CAM results for DCN features across deposit samples
├── visualize_deeplift_deposits.py            #  This script visualizes integrated gradients deposit-level histograms for the study
├── visualize_gradcam_study_area.py            #  This script generates a full-map Grad-CAM anomaly visualization for the study area.
```
research/data/ Geochemical element and geological feature data (.tif) required for sample preparation, including interpolated coordinates
```
