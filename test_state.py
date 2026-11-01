import unittest
from state import Movement, PPEState, seated


def pose(sitting=False):
    points = [[0, 0, 0] for _ in range(17)]
    for i in (5, 6):
        points[i] = [50, 20, 1]
    for i in (11, 12):
        points[i] = [50, 70, 1]
    for i in (13, 14):
        points[i] = [90, 72, 1] if sitting else [50, 120, 1]
    return points


class StateTests(unittest.TestCase):
    def test_seated_and_insufficient_pose(self):
        self.assertTrue(seated(pose(True), 150))
        self.assertFalse(seated(pose(), 150))
        self.assertIsNone(seated(None, 150))

    def test_walk_stand_sit_at_multiple_sampling_rates(self):
        for dt in (0.04, 0.2, 0.4):
            for expected in ('walking', 'standing', 'sitting'):
                tracker = Movement()
                for frame in range(int(4/dt)):
                    t = frame*dt
                    x = 90*t if expected == 'walking' else 0
                    result = tracker.update(1, [x, 0, x+50, 150], pose(expected == 'sitting'), t)
                self.assertEqual(result, expected)

    def test_no_pose_stays_unknown_when_stationary(self):
        tracker = Movement()
        for n in range(30):
            result = tracker.update(1, [0, 0, 50, 150], None, n/10)
        self.assertEqual(result, 'unknown')

    def test_ppe_votes_tolerate_flicker_and_clear(self):
        for dt in (.04,.2):
            state = PPEState()
            for n in range(round(3/dt)):
                result=state.update(1, {'helmet'} if n % 5 != 0 else set(), n*dt)
            self.assertTrue(result['helmet'])
            for n in range(round(3/dt),round(5/dt)):
                result=state.update(1,set(),n*dt)
            self.assertFalse(result['helmet'])
            self.assertIsNone(state.update(1,{'helmet'},20)['helmet'])
            self.assertIsNone(state.update(2,{'helmet'},20)['helmet'])

    def test_long_tracking_gap_restarts_movement(self):
        tracker = Movement()
        for n in range(30):
            tracker.update(1, [n*10, 0, n*10+50, 150], pose(), n/10)
        self.assertEqual(tracker.update(1, [0, 0, 50, 150], None, 20), 'unknown')


if __name__ == '__main__':
    unittest.main()
