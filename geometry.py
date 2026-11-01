import cv2
import numpy as np

def iou(a, b):
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def hivis_score(frame, box, s_min=110, v_min=110):
    """Measure saturated orange-to-lime colour within a central torso crop.

    Returns zero for crops too small to evaluate reliably.
    """
    x1, y1, x2, y2 = box[:4]
    bw, bh = x2 - x1, y2 - y1
    if bw < 8 or bh < 16:
        return 0.0
    torso = frame[max(0, int(y1 + bh * 0.25)):int(y1 + bh * 0.60),
                  max(0, int(x1 + bw * 0.22)):int(x2 - bw * 0.22)]
    if torso.size == 0:
        return 0.0
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], bool)
    for lo, hi in (((8, s_min, v_min), (32, 255, 255)),    # orange -> yellow
                   ((32, s_min, v_min), (52, 255, 255))):  # yellow -> lime
        mask |= cv2.inRange(hsv, lo, hi).astype(bool)
    return float(mask.mean())
