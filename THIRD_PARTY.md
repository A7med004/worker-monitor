# Third-party components

Model weights are downloaded from the upstream publisher, rather than hosted by this project. Model and dependency licences remain with their respective owners.

| Component | Use | Upstream reference |
| --- | --- | --- |
| Ultralytics YOLO26 / YOLOE | Detection, keypoints, PPE, appearance features | [Repository](https://github.com/ultralytics/ultralytics), [licensing](https://www.ultralytics.com/license) |
| Ultralytics assets | Pinned pretrained weight files | [Release](https://github.com/ultralytics/assets/releases/tag/v8.4.0) |
| Ultralytics CLIP | Tokenisation used by the text encoder | [Pinned source](https://github.com/ultralytics/CLIP/tree/7ffa84b3bfa40c42ecc1c77147a855e69cb2dd40) |
| MobileCLIP | Text encoding for PPE prompts | [Apple repository](https://github.com/apple/ml-mobileclip) |
| PyTorch / TorchVision | Inference runtime | [PyTorch](https://pytorch.org/) |
| OpenCV | Video processing and annotations | [OpenCV](https://opencv.org/) |
| Streamlit | Local review interface | [Streamlit](https://streamlit.io/) |

The dependency files pin the inference packages and the CLIP source revision. Consult the upstream licence files for their terms. Refer to `MODELS.md` for asset origins and integrity checksums. This notice does not replace upstream licence terms or assign a licence to the project's own code.
