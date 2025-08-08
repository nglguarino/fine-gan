# FineGAN: Frequency-Trained CNN with Adversarial Refinement for Inpainting

Novel two-stage CNN-GAN architecture for image inpainting, separating frequency-aware coarse reconstruction from adversarial
fine-detail synthesis.

**Authors:** Angelo Guarino, Lisa Milan, Zeynep Tutar

###  📊 CNN Inpainting Evaluation Results (2.5% Mask)

| Metric | Value |
| :--- | :--- |
| **PSNR** | `39.14 ± 1.69 dB` |
| **SSIM** | `0.9898 ± 0.0029` |
| **LPIPS** | `0.0099 ± 0.0026` |
| **FID** | `4.64` |
| **Masked MSE** | `0.0147` |
| **Masked MAE** | `0.1342` |

![CNN Coarse Inpainting](results/cnn_inpainting.png)

---

**For a detailed breakdown of all experiments, methodologies, and additional results:**

[![Project Report](https://img.shields.io/badge/Project_Report-View_on_Notion-000000?logo=notion)](https://ngldatascience.notion.site/inpainting_resDiff-23ae6e287a6880c78713f5fcb844e5f4)