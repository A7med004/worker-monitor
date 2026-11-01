"""Readable annotations: unobstructed video and a separate worker status panel."""
import cv2
import numpy as np

GREEN=(110,215,105)
AMBER=(75,190,255)
GREY=(170,180,195)
WHITE=(235,240,248)


def text(img, value, x, y, colour=WHITE, scale=.5):
    cv2.putText(img,value,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,colour,1,cv2.LINE_AA)


def annotate(frame, rows, time_s):
    h,w=frame.shape[:2]
    top,bottom,side=76,42,360
    height=max(h+top+bottom,720)
    capacity=(height-top-bottom)//116
    height += height%2
    width=w+side+(w+side)%2
    canvas=np.full((height,width,3),(19,24,33),dtype=np.uint8)
    canvas[top:top+h,:w]=frame
    text(canvas,'WORKER MONITOR',18,27,scale=.70)
    text(canvas,f'{time_s:06.1f}s  |  {len(rows)} workers visible',18,53,GREY)
    text(canvas,'PPE OBSERVATIONS',w+18,28,scale=.58)
    text(canvas,'Green: seen   Amber: not seen   Grey: checking',w+18,52,GREY,.37)
    for index,row in enumerate(rows):
        status=row['ppe']
        colour=(AMBER if any(status[k] is False for k in ('helmet','vest')) else
                GREEN if all(status[k] is True for k in ('helmet','vest')) else GREY)
        x1,y1,x2,y2=map(int,row['bbox'])
        x1,x2=max(0,x1),min(w-1,x2)
        y1,y2=max(top,y1+top),min(top+h-1,y2+top)
        cv2.rectangle(canvas,(x1,y1),(x2,y2),colour,2,cv2.LINE_AA)
        tag=f"W-{row['worker']:02d}"
        tx=min(max(0,x1),max(0,w-90)); ty=max(top,y1-25)
        cv2.rectangle(canvas,(tx,ty),(tx+88,ty+23),(25,30,40),-1)
        text(canvas,tag,tx+7,ty+17,colour,.5)
        if index>=capacity:
            continue
        y=top+index*116
        cv2.rectangle(canvas,(w+10,y),(width-10,y+107),(31,38,49),-1)
        cv2.rectangle(canvas,(w+10,y),(w+13,y+107),colour,-1)
        text(canvas,f"{tag}  |  {row['activity'].upper()}",w+22,y+22,WHITE,.51)
        for j,item in enumerate(('helmet','vest')):
            value=status[item]
            label='CHECKING' if value is None else 'SEEN' if value else 'NOT SEEN'
            source=row.get('ppe_sources',{}).get(item)
            if value and source and 'colour' in source:
                label+=' (colour assist)'
            text(canvas,f'{item.title()}: {label}',w+22,y+45+j*21,
                 GREY if value is None else GREEN if value else AMBER,.45)
        advisory='  '.join(f'{k.title()}: {"seen" if status[k] else "?"}' for k in ('gloves','boots'))
        text(canvas,advisory+' [advisory]',w+22,y+91,GREY,.34)
    if len(rows)>capacity:
        text(canvas,f'+ {len(rows)-capacity} workers in the table',18,height-55,AMBER)
    text(canvas,'NOT SEEN means not detected; it does not prove PPE is absent.',18,height-23,GREY,.43)
    return canvas
