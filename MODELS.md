# Model downloads

All assets are pinned to the official [Ultralytics v8.4.0 release](https://github.com/ultralytics/assets/releases/tag/v8.4.0).

Download automatically with `python download_models.py`, or save the following files into `weights/` and run `python download_models.py --check`.

| File | Purpose | Size (MiB) | Official download |
| --- | --- | ---: | --- |
| `yolo26s.pt` | Worker detection | 19.5 | [Download](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt) |
| `yolo26n-pose.pt` | Body keypoints | 7.5 | [Download](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-pose.pt) |
| `yoloe-11s-seg.pt` | PPE detection | 26.5 | [Download](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-11s-seg.pt) |
| `mobileclip_blt.ts` | PPE text encoder | 572.0 | [Download](https://github.com/ultralytics/assets/releases/download/v8.4.0/mobileclip_blt.ts) |
| `yolo26n-cls.pt` | Tracking appearance encoder | 5.5 | [Download](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-cls.pt) |

Total: approximately 631 MiB. No account or API key is required.

## SHA-256 checksums

```text
646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b  yolo26s.pt
eb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9  yolo26n-pose.pt
8e439445c87338b79d9ce21dec109f4621e26df67e94d26ea1a98c1e64dce3e3  yoloe-11s-seg.pt
a67804d1b0f07b8b9a20c1761ec0847f34660f5fa338ec70e8f3fce68ed95e54  mobileclip_blt.ts
0dd6f8dbc448870ac98a3cbb7156f923f7ce21fed3755d4019169ffffd279e81  yolo26n-cls.pt
```

The downloader verifies both the exact byte count and SHA-256 digest before activating a file. The manifest records these values from the official release metadata.

## Upstream references

- [YOLO26 model documentation](https://docs.ultralytics.com/models/yolo26/)
- [YOLOE model documentation](https://docs.ultralytics.com/models/yoloe/)
- [Ultralytics licensing](https://www.ultralytics.com/license)

Keep model binaries out of Git. Manual downloads must use the exact filenames shown above.
