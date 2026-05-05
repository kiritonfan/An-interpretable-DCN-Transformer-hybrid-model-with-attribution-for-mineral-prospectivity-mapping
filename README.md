# An-interpretable-DCN-Transformer-hybrid-model-with-attribution-for-mineral-prospectivity-mapping
Considering mineralization local-global geological features: An interpretable DCN-Transformer hybrid model with attribution for mineral prospectivity mapping

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
├── sample_preparation.py                   #  This script prepares training and validation samples for the mineral prospectivity study
├── ablation_dcn_only.py                   #  This script runs the DCN-only ablation experiment for the study
├── ablation_transformer_only.py                   #  This script runs the Transformer-only ablation experiment for the study
├── ablation_dcn_transformer.py                #  This script runs the DCN-Transformer ablation experiment for the study
├── attribution_guided_model.py                #  This module defines the attribution-guided DCN-Transformer model used in the study
├── dcnv2_model.py               #  This module defines the DCNv2-based feature extraction components used in the study
├── Dense_transformers.py               #  This module defines the dense transformer components used in the study
├── train_attribution_guided.py            #  This script trains and evaluates the attribution-guided model for the study
├── visualize_attention_map.py            #  This script visualizes attention maps produced by the trained model
├── visualize_deeplift_beeswarm.py            #  This script visualizes DeepLIFT feature attributions with beeswarm plots
├── visualize_gradcam_dcn_all_deposits.py            #  This script visualizes Grad-CAM results for DCN features across deposit samples
├── visualize_ig_deposit_histograms.py            #  This script visualizes integrated gradients deposit-level histograms for the study
├── gradcam_full_map.py            #  This script generates a full-map Grad-CAM anomaly visualization for the study area.
```
research/data/ Geochemical element and geological feature data (.tif) required for sample preparation, including interpolated coordinates
