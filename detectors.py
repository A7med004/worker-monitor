"""Project-local person, pose, and PPE model wrappers."""
from pathlib import Path
import tempfile

import yaml
from ultralytics import YOLO, YOLOE

from download_models import ensure_models

TRACKER = Path(__file__).with_name('tracker.yaml')
PPE_PROMPTS = ["hard hat", "high-visibility vest", "gloves", "boots"]
PPE_NAME_MAP = {"hard hat": "helmet", "high-visibility vest": "vest",
                "gloves": "gloves", "boots": "boots"}
PPE_CONF = {"helmet": 0.30, "vest": 0.10, "gloves": 0.30, "boots": 0.30}


def load_ppe(root):
    """Load the text encoder by absolute path without changing the working directory."""
    from ultralytics.nn.text_model import MobileCLIPTS
    model = YOLOE(str(root / 'yoloe-11s-seg.pt'))
    model.model.clip_model = MobileCLIPTS(
        device=next(model.model.parameters()).device,
        weight=str(root / 'mobileclip_blt.ts'),
    )
    try:
        embeddings = model.model.get_text_pe(PPE_PROMPTS, cache_clip_model=True)
        model.set_classes(PPE_PROMPTS, embeddings)
    finally:
        del model.model.clip_model
    return model


class Detectors:
    def __init__(self, device=None):
        self.root = ensure_models()
        self.person = YOLO(str(self.root / 'yolo26s.pt'))
        self.ppe = load_ppe(self.root)
        self._ppe_map = PPE_NAME_MAP
        self.pose = YOLO(str(self.root / 'yolo26n-pose.pt'))
        if device is not None and str(device).isdigit():
            device = f'cuda:{device}'
        if device:
            for model in (self.person, self.ppe, self.pose):
                model.to(device)
        # An explicit ReID path prevents hidden downloads into the caller's directory.
        self._tracker_dir = tempfile.TemporaryDirectory(prefix='worker-monitor-tracker-')
        config = yaml.safe_load(TRACKER.read_text(encoding='utf-8-sig'))
        config['model'] = str(self.root / 'yolo26n-cls.pt')
        self.tracker = Path(self._tracker_dir.name) / 'tracker.yaml'
        self.tracker.write_text(yaml.safe_dump(config), encoding='utf-8')

    def _ppe_name(self, i):
        n = self.ppe.names[int(i)]
        return self._ppe_map.get(n, n) if self._ppe_map else n

    def detect_persons(self, frame, conf=0.3):
        """Return [(track_id, x1, y1, x2, y2, conf)] using BoT-SORT + ReID.
        Must be called on EVERY frame for stable IDs."""
        r = self.person.track(frame, persist=True, tracker=str(self.tracker),
                              classes=[0], conf=conf, verbose=False)[0]
        out = []
        if r.boxes is not None:
            for b in r.boxes:
                tid = int(b.id.item()) if b.id is not None else -1
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                out.append((tid, x1, y1, x2, y2, float(b.conf[0])))
        return out

    def detect_ppe(self, frame, conf=0.25):
        """Return {class_name: [(x1,y1,x2,y2,conf)]} from the PPE model."""
        r = self.ppe.predict(frame, conf=conf, verbose=False)[0]
        out = {}
        if r.boxes is not None:
            for b in r.boxes:
                name = self._ppe_name(b.cls[0])
                out.setdefault(name, []).append((*b.xyxy[0].tolist(), float(b.conf[0])))
        return out

    def detect_ppe_per_person(self, frame, persons, conf=None, pad=0.12,
                              imgsz=384, class_conf=None):
        """Anatomically filter crop detections, retry weak PPE at 640px.

        last_ppe_raw contains actual model scores; colour evidence is handled
        separately and never inserted as a model confidence.
        """
        from evidence import belongs_to_worker
        H, W = frame.shape[:2]
        crops, metadata = [], []
        for person in persons:
            tid, x1, y1, x2, y2, _ = person
            bw, bh = x2-x1, y2-y1
            left, top = max(0, int(x1-pad*bw)), max(0, int(y1-pad*bh))
            right, bottom = min(W, int(x2+pad*bw)), min(H, int(y2+pad*bh))
            if right-left < 8 or bottom-top < 16:
                continue
            crops.append(frame[top:bottom, left:right])
            metadata.append((tid, person[1:5], left, top))
        floor = 0.025 if conf is None else conf
        thresholds = PPE_CONF if class_conf is None else class_conf
        raw = {tid: {} for tid, *_ in metadata}
        def collect(results, entries):
            for result, (tid, worker_box, left, top) in zip(results, entries):
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    name, score = self._ppe_name(box.cls[0]), float(box.conf[0])
                    bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                    absolute = (bx1+left, by1+top, bx2+left, by2+top)
                    if belongs_to_worker(name, absolute, worker_box):
                        raw[tid][name] = max(score, raw[tid].get(name, 0.0))
        if crops:
            collect(self.ppe.predict(crops, conf=floor, imgsz=imgsz, verbose=False), metadata)
            retry = [i for i, (tid, *_) in enumerate(metadata)
                     if raw[tid].get('vest', 0) < thresholds['vest']
                     or raw[tid].get('helmet', 0) < thresholds['helmet']]
            if retry:
                collect(self.ppe.predict([crops[i] for i in retry], conf=floor,
                                        imgsz=640, verbose=False),
                        [metadata[i] for i in retry])
        self.last_ppe_raw = raw
        return {tid: {name: score for name, score in scores.items()
                      if score >= thresholds.get(name, 1.0)}
                for tid, scores in raw.items()}

    def detect_poses(self, frame, conf=0.3):
        """Return [(x1,y1,x2,y2,conf, kpts[17,3])] for posed persons."""
        r = self.pose.predict(frame, conf=conf, verbose=False)[0]
        out = []
        if r.boxes is not None:
            for b, k in zip(r.boxes, r.keypoints.data):
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                out.append((x1, y1, x2, y2, float(b.conf[0]), k.cpu().numpy()))
        return out
