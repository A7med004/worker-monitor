"""Keep model confidence and colour evidence separate."""
import cv2
import numpy as np


def belongs_to_worker(item, detected, person):
    x1, y1, x2, y2 = person
    w, h = x2-x1, y2-y1
    if w <= 0 or h <= 0:
        return False
    cx, cy = (detected[0]+detected[2])/2, (detected[1]+detected[3])/2
    rx, ry = (cx-x1)/w, (cy-y1)/h
    limits = {'helmet': (-.15, .30), 'vest': (.12, .72),
              'gloves': (.20, .90), 'boots': (.65, 1.12)}
    low, high = limits.get(item, (0, 1))
    return -.10 <= rx <= 1.10 and low <= ry <= high


def vest_colour(frame, box):
    x1,y1,x2,y2 = box
    w,h = x2-x1,y2-y1
    if w < 24 or h < 60:
        return dict(fraction=0.0, bands=False)
    H,W = frame.shape[:2]
    torso = frame[max(0,int(y1+.18*h)):min(H,int(y1+.62*h)),
                  max(0,int(x1+.10*w)):min(W,int(x2-.10*w))]
    if not torso.size:
        return dict(fraction=0.0, bands=False)
    hsv = cv2.cvtColor(torso,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([8,110,110]),np.array([52,255,255])) > 0
    # Open vests appear as two vertical fluorescent strips around a dark shirt.
    third = max(1,mask.shape[1]//3)
    left,right = mask[:,:third], mask[:,-third:]
    vertical = lambda area: sum(float(part.mean()) >= .08
                               for part in np.array_split(area,4,axis=0) if part.size) >= 3
    bands = (left.mean() >= .15 and right.mean() >= .15
             and vertical(left) and vertical(right))
    return dict(fraction=round(float(mask.mean()),4),bands=bool(bands))


def combine_ppe(scores, colour, assist=True):
    thresholds = {'helmet':.30, 'vest':.10, 'gloves':.30, 'boots':.30}
    detected = {item:score for item,score in scores.items()
                if score >= thresholds.get(item,1)}
    sources = {item:'model' for item in detected}
    if assist and 'vest' not in detected:
        supported = scores.get('vest',0) >= .04 and colour['fraction'] >= .10 and colour['bands']
        strong_bands = colour['fraction'] >= .18 and colour['bands']
        if supported or strong_bands:
            # No synthetic model score: a colour observation is a separate source.
            detected['vest'] = scores.get('vest',0)
            sources['vest'] = 'model+colour' if supported else 'colour'
    return detected, sources


def unobservable_ppe(box, people, tid, frame_shape):
    """Avoid negative observations for tiny workers or covered body regions."""
    x1,y1,x2,y2=box
    w,h=x2-x1,y2-y1
    if h<60 or w<16:
        return {'helmet','vest','gloves','boots'}
    unknown=set()
    if y2>=frame_shape[0]-3:
        unknown.add('boots')
    for item,low,high in [('helmet',0,.25),('vest',.18,.62)]:
        region=(x1+.1*w,y1+low*h,x2-.1*w,y1+high*h)
        area=max(1,(region[2]-region[0])*(region[3]-region[1]))
        for other_id,ox1,oy1,ox2,oy2,_ in people:
            if other_id==tid or oy2<y2-.05*h:
                continue
            intersection=max(0,min(region[2],ox2)-max(region[0],ox1))*max(0,min(region[3],oy2)-max(region[1],oy1))
            if intersection/area>.25:
                unknown.add(item)
    return unknown
