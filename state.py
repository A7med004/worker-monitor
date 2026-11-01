"""Time-based movement and PPE smoothing. No model dependencies."""
import math

ITEMS = ('helmet', 'vest', 'gloves', 'boots')


def seated(kpts, height):
    if kpts is None:
        return None
    def midpoint(indices):
        points = [kpts[i] for i in indices if kpts[i][2] > 0.3]
        if not points:
            return None
        return tuple(sum(p[j] for p in points) / len(points) for j in (0, 1))
    shoulders, hips, knees = [midpoint(i) for i in ((5, 6), (11, 12), (13, 14))]
    if shoulders is None or hips is None or knees is None:
        return None
    # Horizontal torsos cannot reliably be classified into these three states.
    if abs(hips[1] - shoulders[1]) < abs(hips[0] - shoulders[0]) * 0.6:
        return None
    dx, dy = knees[0] - hips[0], knees[1] - hips[1]
    return abs(dy) < abs(dx) * 0.6 and abs(dy) < height * 0.15


class Movement:
    def __init__(self):
        self.history = {}

    def update(self, tid, box, kpts, time_s):
        x1, y1, x2, y2 = box
        center = ((x1+x2)/2, (y1+y2)/2)
        height = max(y2-y1, 20)
        old = self.history.get(tid)
        if old is None or time_s-old['time'] > 1.0:
            old = dict(center=center, time=time_s, speed=0., current='unknown',
                       candidate=None, since=time_s)
            self.history[tid] = old
        dt = time_s-old['time']
        if dt > 0:
            speed = math.dist(center, old['center']) / height / dt
            old['speed'] += (1-math.exp(-dt/0.6)) * (speed-old['speed'])
        sitting = seated(kpts, height)
        raw = ('sitting' if sitting else 'walking' if old['speed'] >= 0.25
               else 'standing' if sitting is False else 'unknown')
        if raw != old['candidate']:
            old.update(candidate=raw, since=time_s)
        if time_s-old['since'] >= 0.6:
            old['current'] = raw
        old.update(center=center, time=time_s)
        return old['current']


class PPEState:
    """Time-weighted votes: tolerate intermittent hits, expire stale identities."""
    def __init__(self):
        self.history = {}

    def update(self, tid, detected, time_s, unobservable=()):
        from collections import deque
        state = self.history.get(tid)
        if state is None or time_s-state['last'] > 1.0:
            state = {'last': time_s, 'samples': deque(),
                     'values': {item: None for item in ITEMS}}
            self.history[tid] = state
        samples = state['samples']
        observed = {item: None if item in unobservable and item not in detected else item in detected for item in ITEMS}
        samples.append((time_s, observed))
        while len(samples) > 1 and samples[1][0] < time_s-1.5:
            samples.popleft()
        state['last'] = time_s
        for item in ITEMS:
            positive, total = 0.0, 0.0
            for (start, items), (end, _) in zip(samples, list(samples)[1:]):
                duration = max(0, min(end-start, 0.5, end-max(start, time_s-1.5)))
                if items[item] is None:
                    continue
                total += duration
                if items[item]:
                    positive += duration
            if observed[item] is None or total < 1.0:
                state['values'][item] = None
            elif positive/total >= 0.6:
                state['values'][item] = True
            elif positive/total <= 0.2:
                state['values'][item] = None if item in detected else False
            else:
                state['values'][item] = None
        return dict(state['values'])
