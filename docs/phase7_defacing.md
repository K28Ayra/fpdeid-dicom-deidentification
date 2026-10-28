# Phase 7: Brain-protected defacing (optional)

Code: [`deface.py`](../src/fpdeid/deface.py) · tests: [`test_deface.py`](../tests/test_deface.py) ·
results in [`results/phase7/`](../results/phase7/)

MIDI-B has no head scans, so nothing here changes the MIDI-B numbers. I used 30 T1 scans from IXI (CC BY-SA 3.0):
10 to choose settings, 20 to test.

The method: SynthStrip (Hoopes et al., NeuroImage 2022) gives the brain mask; the mask is dilated by 10 mm and those voxels
are never changed. The face region is everything in front of the middle of the brain, below the height of the frontal pole
plus 15 mm, and above the frontal pole minus 120 mm (so the neck stays). Face region minus protected voxels is set to the
image minimum. A DICOM head series (found by *Body Part Examined* or words like HEAD/BRAIN) is stacked by slice position,
defaced, and only changed pixels are written back.

## Checking for faces

Each head surface is rendered from 8 directions with a depth-shaded projection in numpy, and two OpenCV face detectors
look at the renders: YuNet (2023) and Ultra-Light-Fast RFB-320 (2019). OpenCV 5 dropped the Haar cascades and Caffe models
I first wanted, and going back to OpenCV 4 might have changed the Phase 4 OCR, so I used these two ONNX models.

Both detectors, trained on photos, fired on renders that show no face, even on the back of a head. Tuning thresholds on the
defaced images would have made the test meaningless. The thresholds are set just above each detector's highest score on
back-of-head views of the original scans, and a face counts only if both detectors find it. With that, the original scans
give faces in 100% of front views, 97% at ±30° and 0% from behind.

Before switching to the two-detector rule I tried protecting the brain's convex hull instead of its exact shape, expecting
a smoother surface to trigger fewer detections. It helped only a little (16 instead of 22 flagged views at a 10 mm
margin, under the old single-detector rule) and was worse at 5 mm, so I dropped it.

## Results, 20 test scans

| | FP-DeID | quickshear | fixed box | original |
|---|---|---|---|---|
| Views with a face, both detectors (of 160) | 7 | 36 | 11 | 88 |
| Brain voxels changed | 0 | 0 | 502,124 | |
| Tissue removed (ml) | 918 | 272 | 887 | |
| Skull-strip Dice after defacing | 0.994 | 0.996 | 0.988 | |
| Brain volume change | 0.70% | 0.42% | 1.89% | |
| Tissue segmentation Dice | 0.986 | 0.990 | 0.977 | |

quickshear is Schimke & Hale's plane-cut defacer (version 1.2.0). The box zeroes the lower front 45% x 45% of the field of
view. I looked at all remaining flagged views ([`remaining_detections.png`](../results/phase7/remaining_detections.png)):
for FP-DeID it is the rim of protected tissue around the brain, for quickshear the head outline, ears and sinus hollows on
the cut. None shows eyes, nose and mouth.

![Renders after defacing](../results/phase7/renders.png)

![Sagittal slice after defacing](../results/phase7/sagittal.png)

Two targets from the plan are missed: brain volume change (0.70%, target < 0.5%) and tissue Dice (0.986, target ≥ 0.99).
The brain voxels are identical; what changes is SynthStrip's mask when it runs again on a head without a face. quickshear
removes much less tissue and moves the mask less, but leaves five times more flagged views. I have not found a setting that
gets both. Brain masks should be made before defacing.

The settings came from 18 combinations on the tune scans ([`tuning.csv`](../results/phase7/tuning.csv)). Margin 10 mm,
cut 15 mm above the pole, chin limit 120 mm gave 3 flagged views of 80 and 953 ml removed; a 20 mm margin removed less
(726 ml) but gave 4.

## Smaller problems

- IXI only serves one 4.8 GB T1 archive at about 100 KB/s. It is a plain tar, so 16 range requests for the first 240 MB
  were enough for 30 complete scans ([`download_ixi.sh`](../scripts/download_ixi.sh)). The SynthStrip weights had
  arrived truncated the first time and were fetched the same way.
- On the GPU, two SynthStrip runs on the same scan differed in about 30 voxels. On the CPU they are identical and take the
  same 8 s, so the CPU is used.
- My first DICOM check wrote the IXI scan as a straight series while the original is slightly oblique, so the two versions
  had different geometry. Compared on the same geometry, DICOM and volume defacing agree on every voxel, and all 256 headers
  are unchanged.
- For CT, 0 is water. The face is filled with the image minimum, which is the background for MR and CT.
- The full notebook takes about 6 hours, almost all of it rotating volumes for the renders on one CPU core.
