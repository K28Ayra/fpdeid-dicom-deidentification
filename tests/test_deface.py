"""Tests for Phase 7 defacing on a synthetic head (a big ball with a smaller 'brain' ball). No SynthStrip, no GPU."""

import numpy as np
from pydicom.dataset import Dataset

from fpdeid import deface

ZOOMS = np.array([1.0, 1.0, 1.0], dtype=np.float32)


def synthetic_head():
    shape = (80, 90, 90)
    x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
    head = (x - 40) ** 2 + (y - 45) ** 2 + (z - 45) ** 2 <= 38 ** 2
    brain = (x - 40) ** 2 + (y - 42) ** 2 + (z - 52) ** 2 <= 25 ** 2
    volume = np.where(head, 100.0, 0.0).astype(np.float32)
    volume[brain] = 200.0
    return volume, head, brain


def test_brain_and_its_margin_are_never_changed():
    volume, head, brain = synthetic_head()
    defaced, removed = deface.deface_fpdeid(volume, brain, ZOOMS, margin_mm=5.0, above_frontal_pole_mm=10.0)
    assert (defaced[brain] == volume[brain]).all()
    protected = np.zeros_like(brain)
    from scipy import ndimage
    protected = ndimage.binary_dilation(brain, structure=deface.ball(5.0, ZOOMS))
    assert not (removed & protected).any()
    assert (removed & head).sum() > 0                           # some face tissue was removed


def test_face_region_is_front_and_low():
    volume, head, brain = synthetic_head()
    region = deface.face_region(brain, ZOOMS, above_frontal_pole_mm=10.0)
    coordinates = np.argwhere(region)
    assert coordinates[:, 1].min() >= 42                        # in front of the middle of the brain
    assert coordinates[:, 2].max() < 52 + 10 + 1                # below the cut height


def test_box_method_damages_the_brain_on_this_head():
    volume, head, brain = synthetic_head()
    defaced, removed = deface.deface_box(volume, front_share=0.5, low_share=0.6)
    assert (defaced[brain] != volume[brain]).any()


def test_head_mask_render_and_dice():
    volume, head, brain = synthetic_head()
    mask = deface.head_mask(volume)
    assert deface.dice(mask, head) > 0.95
    picture = deface.render(mask, yaw_degrees=30)
    assert picture.dtype == np.uint8 and picture.max() > 100
    assert deface.dice(brain, brain) == 1.0


def test_tissue_classes_inside_the_brain_only():
    volume, head, brain = synthetic_head()
    noisy = volume + np.random.default_rng(0).normal(0, 5, volume.shape).astype(np.float32)
    classes = deface.tissue_classes(noisy, brain)
    assert (classes[~brain] == 0).all()
    assert set(np.unique(classes[brain])) <= {1, 2, 3}


def test_head_series_detection():
    ds = Dataset()
    ds.Modality = "MR"
    ds.BodyPartExamined = "BRAIN"
    assert deface.is_head_series(ds)
    ds.BodyPartExamined = "BREAST"
    ds.SeriesDescription = "T1 axial"
    assert not deface.is_head_series(ds)


def test_dicom_series_path_defaces_the_front_like_the_volume_path(tmp_path, monkeypatch):
    """The DICOM path must find the same face voxels as the volume path (catches left/right and front/back mix-ups)."""
    volume, head, brain = synthetic_head()
    affine = np.eye(4)
    # SynthStrip is replaced by the known synthetic brain (the volume path and the DICOM path both see RAS)
    monkeypatch.setattr(deface, "brain_mask", lambda vol, aff, use_gpu=True: vol >= 150)
    files = deface.write_series(volume, affine, str(tmp_path / "input"))
    outputs = [path.replace("input", "output") for path in files]
    (tmp_path / "output").mkdir()
    result = deface.deface_dicom_series(files, outputs, margin_mm=5.0, above_frontal_pole_mm=10.0)
    assert result["slices"] == volume.shape[2] and result["slices changed"] > 0

    pairs_in, volume_in, affine_in = deface.series_volume(files)
    pairs_out, volume_out, affine_out = deface.series_volume(outputs)
    changed = (volume_in != volume_out)[::-1, ::-1, :]          # back to RAS (the writer reverses x and y)
    direct, removed = deface.deface_fpdeid(volume, brain, ZOOMS, margin_mm=5.0, above_frontal_pole_mm=10.0)
    expected = removed & (volume != volume.min())
    assert (changed == expected).all()
    assert np.argwhere(changed)[:, 1].min() >= 42               # only the front of the head changed
