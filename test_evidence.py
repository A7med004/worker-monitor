import unittest
import cv2
import numpy as np
from evidence import vest_colour, combine_ppe, belongs_to_worker, unobservable_ppe
from annotation import annotate

class EvidenceTests(unittest.TestCase):
    def test_anatomical_filter(self):
        self.assertTrue(belongs_to_worker('helmet',(30,0,70,25),(0,0,100,200)))
        self.assertFalse(belongs_to_worker('helmet',(30,120,70,150),(0,0,100,200)))
        self.assertFalse(belongs_to_worker('vest',(110,40,180,130),(0,0,100,200)))

    def test_colour_bands_support_open_vest_and_can_be_disabled(self):
        frame=np.zeros((200,100,3),np.uint8)
        frame[36:124,12:30]=(0,255,255)
        frame[36:124,70:88]=(0,255,255)
        colour=vest_colour(frame,(0,0,100,200))
        self.assertTrue(colour['bands'])
        items,sources=combine_ppe({},colour)
        self.assertIn('vest',items)
        self.assertEqual(sources['vest'],'colour')
        self.assertNotIn('vest',combine_ppe({},colour,False)[0])
        # Blue clothing never provides fluorescent evidence.
        frame[:]=(255,0,0)
        self.assertFalse(vest_colour(frame,(0,0,100,200))['bands'])
        self.assertNotIn('vest',combine_ppe({},vest_colour(frame,(0,0,100,200)))[0])

    def test_one_colour_patch_is_not_a_vest(self):
        frame=np.zeros((200,100,3),np.uint8)
        frame[36:65,10:90]=(0,255,255)
        colour=vest_colour(frame,(0,0,100,200))
        self.assertFalse(colour['bands'])
        self.assertNotIn('vest',combine_ppe({'vest':.06},colour)[0])

    def test_occlusion_is_unknown_not_negative_evidence(self):
        from state import PPEState
        box=[0,0,100,200]
        hidden=unobservable_ppe(box,[(2,0,30,100,230,.9)],1,(480,640,3))
        self.assertIn('vest',hidden)
        state=PPEState()
        for n in range(30):
            status=state.update(1,set(),n/10,hidden)
        self.assertIsNone(status['vest'])

    def test_annotation_keeps_fixed_video_dimensions(self):
        frame=np.zeros((480,586,3),np.uint8)
        row=dict(worker=1,bbox=[100,50,200,300],activity='standing',
                 ppe={k:None for k in ('helmet','vest','gloves','boots')})
        self.assertEqual(annotate(frame,[],0).shape,annotate(frame,[row]*20,1).shape)

if __name__=='__main__':
    unittest.main()
