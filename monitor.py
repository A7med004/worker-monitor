"""Worker tracking, per-worker PPE, and walking/standing/sitting only."""
import argparse
import json
import math
from pathlib import Path

import cv2

from detectors import Detectors
from geometry import iou
from evidence import vest_colour, combine_ppe, unobservable_ppe
from annotation import annotate
from state import Movement, PPEState, ITEMS


def run(source, output='output', stride=1, device=None, preview=False,
        on_frame=None, max_frames=None, colour_assist=True):
    if stride < 1:
        raise ValueError('stride must be at least 1')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(source)
    writer = None
    count = 0
    seen = set()
    try:
        if not capture.isOpened():
            raise ValueError(f'Cannot open video source: {source}')
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            fps = 25.0
        detectors = Detectors(device=device)  # fresh tracking state for every run
        movement, ppe = Movement(), PPEState()
        with (output/'workers.jsonl').open('w', encoding='utf-8') as log:
            frame_index = -1
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_index += 1
                # Feed every frame to the tracker, including skipped inference frames.
                people = [p for p in detectors.detect_persons(frame) if p[0] >= 0]
                if frame_index % stride:
                    continue
                time_s = frame_index/fps
                detected = detectors.detect_ppe_per_person(frame, people)
                poses = detectors.detect_poses(frame)
                # Greedy one-to-one pose assignment avoids crediting one pose twice.
                matches, used_people, used_poses = {}, set(), set()
                pairs = sorted(((iou(p[1:5], q[:4]), i, j)
                                for i, p in enumerate(people)
                                for j, q in enumerate(poses)), reverse=True)
                for overlap, i, j in pairs:
                    if overlap > 0.3 and i not in used_people and j not in used_poses:
                        matches[i] = poses[j][5]
                        used_people.add(i)
                        used_poses.add(j)
                rows = []
                for i, (tid, x1, y1, x2, y2, confidence) in enumerate(people):
                    box = [x1, y1, x2, y2]
                    raw_scores = detectors.last_ppe_raw.get(tid, {})
                    colour = vest_colour(frame, box)
                    items, sources = combine_ppe(raw_scores, colour, colour_assist)
                    unclear = unobservable_ppe(box, people, tid, frame.shape)
                    status = ppe.update(tid, items, time_s, unclear)
                    activity = movement.update(tid, box, matches.get(i), time_s)
                    seen.add(tid)
                    row = dict(frame=frame_index, time_s=round(time_s, 3),
                               worker=tid, bbox=box, confidence=confidence,
                               activity=activity, ppe=status, ppe_scores=raw_scores,
                               ppe_sources=sources, vest_colour=colour, ppe_unobservable=sorted(unclear))
                    rows.append(row)
                    log.write(json.dumps(row)+'\n')
                frame = annotate(frame, rows, time_s)
                if writer is None:
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(str(output/'annotated.mp4'),
                                             cv2.VideoWriter_fourcc(*'mp4v'),
                                             fps/stride, (width, height))
                    if not writer.isOpened():
                        raise RuntimeError('Cannot create output video')
                writer.write(frame)
                count += 1
                if on_frame:
                    on_frame(frame, rows)
                if preview:
                    cv2.imshow('Worker Monitor - Q to stop', frame)
                    if cv2.waitKey(1) & 0xff == ord('q'):
                        break
                if max_frames and count >= max_frames:
                    break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if preview:
            cv2.destroyAllWindows()
    summary = dict(source=str(source), processed_frames=count,
                   worker_track_ids=sorted(seen), stride=stride, colour_assist=colour_assist,
                   note='Track IDs can change after occlusion; counts are not verified unique people.')
    (output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video')
    parser.add_argument('--webcam', action='store_true')
    parser.add_argument('--out', default=str(Path(__file__).parent/'output'))
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--device', default=None)
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--no-colour-assist', action='store_true')
    args = parser.parse_args()
    if not args.webcam and not args.video:
        parser.error('Choose --video PATH or --webcam')
    print(run(0 if args.webcam else args.video, args.out, args.stride,
              args.device, args.preview, max_frames=args.max_frames,
              colour_assist=not args.no_colour_assist))
