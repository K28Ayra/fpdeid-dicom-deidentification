"""Phase 7 (optional): brain-protected defacing of head MRI."""

import os
import subprocess
import sys
import tempfile

import cv2
import nibabel as nib
import numpy as np
from scipy import ndimage

from fpdeid import paths

SYNTHSTRIP = os.path.join(paths.PROJECT, "tools", "synthstrip", "mri_synthstrip")
SYNTHSTRIP_WEIGHTS = os.path.join(paths.PROJECT, "tools", "synthstrip", "synthstrip.1.pt")
YUNET = os.path.join(paths.PROJECT, "tools", "face", "face_detection_yunet_2023mar.onnx")
ULTRALIGHT = os.path.join(paths.PROJECT, "tools", "face", "version-RFB-320.onnx")
TEMP = os.path.join(paths.DATA, "tmp")          # temporary files on /home (the root disk is nearly full)


# Reading and writing volumes

def load_ras(path):
    """NIfTI -> (volume as float32 in RAS orientation, affine, voxel size in mm)."""
    image = nib.as_closest_canonical(nib.load(path))
    volume = np.asarray(image.dataobj, dtype=np.float32)
    zooms = np.array(image.header.get_zooms()[:3], dtype=np.float32)
    return volume, image.affine, zooms


def save_nifti(volume, affine, path):
    nib.save(nib.Nifti1Image(volume.astype(np.float32), affine), path)


# 1. Brain mask (SynthStrip) and head mask

def brain_mask(volume, affine, use_gpu=False):
    """Run SynthStrip on a volume. Returns a boolean mask with the same shape.

    The CPU is used by default: it is as fast as the GPU here (about 8 s per scan) and gives exactly the same mask
    every time (on the GPU two runs differ in a few dozen voxels, which would make the output change from run to run).
    """
    os.makedirs(TEMP, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=TEMP) as folder:
        input_path = os.path.join(folder, "input.nii.gz")
        mask_path = os.path.join(folder, "mask.nii.gz")
        save_nifti(volume, affine, input_path)
        command = [sys.executable, SYNTHSTRIP, "-i", input_path, "-m", mask_path, "--model", SYNTHSTRIP_WEIGHTS]
        if use_gpu:
            command.append("-g")
        subprocess.run(command, check=True, capture_output=True)
        mask, mask_affine, zooms = load_ras(mask_path)
    return mask > 0.5


def head_mask(volume):
    """Everything that is not air: a threshold at 10% of the bright tissue, holes filled, the largest piece kept."""
    bright = np.percentile(volume, 99)
    mask = volume > 0.1 * bright
    mask = ndimage.binary_opening(mask, iterations=1)
    labels, count = ndimage.label(mask)
    if count > 1:
        sizes = ndimage.sum(mask, labels, range(1, count + 1))
        mask = labels == (1 + int(np.argmax(sizes)))
    filled = np.zeros_like(mask)
    for z in range(mask.shape[2]):
        filled[:, :, z] = ndimage.binary_fill_holes(mask[:, :, z])
    return filled


def ball(radius_mm, zooms):
    """A 3D ball-shaped structuring element with the given radius in mm."""
    radius = np.maximum(1, np.round(radius_mm / zooms)).astype(int)
    x, y, z = np.ogrid[-radius[0]:radius[0] + 1, -radius[1]:radius[1] + 1, -radius[2]:radius[2] + 1]
    inside = (x * zooms[0]) ** 2 + (y * zooms[1]) ** 2 + (z * zooms[2]) ** 2 <= radius_mm ** 2
    return inside


# 2-4. FP-DeID brain-protected defacing

def face_region(brain, zooms, above_frontal_pole_mm=15.0, below_frontal_pole_mm=None):
    """Voxels in front of the middle of the brain, between chin level and eyebrow level.

    Eyebrow level = the height of the front-most point of the brain (the frontal pole) + above_frontal_pole_mm.
    Chin level    = the same point - below_frontal_pole_mm (None = no lower limit: the whole front of the neck).
    """
    coordinates = np.argwhere(brain)
    middle_y = coordinates[:, 1].mean()
    front_y = coordinates[:, 1].max()
    at_front = coordinates[coordinates[:, 1] >= front_y - round(5 / zooms[1])]
    pole_z = at_front[:, 2].mean()
    cut_z = pole_z + above_frontal_pole_mm / zooms[2]
    low_z = 0
    if below_frontal_pole_mm is not None:
        low_z = max(0, int(np.floor(pole_z - below_frontal_pole_mm / zooms[2])))
    region = np.zeros(brain.shape, dtype=bool)
    region[:, int(np.ceil(middle_y)):, low_z:int(np.floor(cut_z))] = True
    return region


def deface_fpdeid(volume, brain, zooms, margin_mm=10.0, above_frontal_pole_mm=15.0, below_frontal_pole_mm=None):
    """Returns (defaced volume, removed-voxel mask). Voxels within margin_mm of the brain are never changed."""
    protected = ndimage.binary_dilation(brain, structure=ball(margin_mm, zooms))
    remove = face_region(brain, zooms, above_frontal_pole_mm, below_frontal_pole_mm) & ~protected
    defaced = volume.copy()
    defaced[remove] = volume.min()          # background (0 for MR, air for CT)
    return defaced, remove


# Methods to compare with

def deface_box(volume, front_share=0.45, low_share=0.45):
    """The simplest idea: zero a fixed box (front 45% x lower 45% of the field of view). Knows nothing about the brain."""
    remove = np.zeros(volume.shape, dtype=bool)
    y_start = int(volume.shape[1] * (1 - front_share))
    z_end = int(volume.shape[2] * low_share)
    remove[:, y_start:, :z_end] = True
    defaced = volume.copy()
    defaced[remove] = 0
    return defaced, remove


def deface_quickshear(volume, brain, buffer_voxels=10):
    """quickshear (Schimke & Hale, 2011): cut along a plane tangent to the brain's convex hull (side view)."""
    import quickshear
    defaced_image = quickshear.quickshear(nib.Nifti1Image(volume, np.eye(4)),
                                          nib.Nifti1Image(brain.astype(np.uint8), np.eye(4)), buff=buffer_voxels)
    defaced = np.asarray(defaced_image.dataobj, dtype=np.float32)
    remove = (volume != 0) & (defaced == 0)
    return defaced, remove


# Face check: render the head surface and run face detectors

def isotropic(mask, zooms, size_mm=1.0):
    factors = zooms / size_mm
    return ndimage.zoom(mask.astype(np.uint8), factors, order=0) > 0


def render(head, yaw_degrees=0.0, pitch_degrees=0.0):
    """Depth-shaded picture of the head surface seen from the front (after turning the head by yaw/pitch).

    For every pixel, the first head voxel along the viewing direction is found; its depth gives a relief image
    that is shaded with light from the viewer. head must be isotropic (1 mm voxels), RAS.
    """
    turned = head
    if yaw_degrees != 0:
        turned = ndimage.rotate(turned.astype(np.uint8), yaw_degrees, axes=(0, 1), reshape=False, order=0) > 0
    if pitch_degrees != 0:
        turned = ndimage.rotate(turned.astype(np.uint8), pitch_degrees, axes=(1, 2), reshape=False, order=0) > 0
    # viewer in front (high y), looking backwards: first voxel from the front
    depth_axis = turned.shape[1]
    reversed_mask = turned[:, ::-1, :]
    hit = reversed_mask.any(axis=1)
    first = np.argmax(reversed_mask, axis=1).astype(np.float32)
    depth = np.where(hit, first, depth_axis).astype(np.float32)
    depth = ndimage.gaussian_filter(depth, 1.0)
    gradient_x = np.gradient(depth, axis=0)
    gradient_z = np.gradient(depth, axis=1)
    normal_length = np.sqrt(gradient_x ** 2 + gradient_z ** 2 + 1.0)
    light = 1.0 / normal_length                         # surfaces facing the viewer are bright
    near = 1.0 - depth / depth_axis                     # nearer is brighter
    image = np.where(hit, 255 * (0.65 * light + 0.35 * near), 0)
    image = np.clip(image, 0, 255).astype(np.uint8)
    # array (x, z) -> picture rows = top to bottom (z from high to low), columns = viewer's left to right
    picture = image.T[::-1, ::-1]
    return np.ascontiguousarray(picture)


VIEWS = [(0, 0), (-30, 0), (30, 0), (-60, 0), (60, 0), (-90, 0), (90, 0), (0, -20)]   # 8 views (yaw, pitch)


class FaceDetectors:
    """Two independent CNN face detectors that run in OpenCV: YuNet (2023) and Ultra-Light-Fast RFB-320 (2019).

    Both were trained on photos, so on a rendered head they sometimes fire on shapes that are not a face (an ear, a
    fold of tissue). That is why the thresholds are calibrated (see calibrate) and a face only counts when BOTH agree.
    """

    def __init__(self, yunet_score=0.5, ultralight_score=0.8):
        self.yunet = cv2.FaceDetectorYN.create(YUNET, "", (320, 320), score_threshold=0.01)
        self.ultralight = cv2.dnn.readNetFromONNX(ULTRALIGHT)
        self.yunet_score = yunet_score
        self.ultralight_score = ultralight_score

    def scores(self, picture):
        """The best face score of each detector for one grey picture (0 = nothing found)."""
        size = 400
        scale = size / max(picture.shape)
        resized = cv2.resize(picture, (int(picture.shape[1] * scale), int(picture.shape[0] * scale)))
        padded = np.zeros((size + 100, size + 100), dtype=np.uint8)
        padded[50:50 + resized.shape[0], 50:50 + resized.shape[1]] = resized
        colour = cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)

        self.yunet.setInputSize((colour.shape[1], colour.shape[0]))
        found, faces = self.yunet.detect(colour)
        yunet_best = 0.0
        if faces is not None and len(faces) > 0:
            yunet_best = float(faces[:, -1].max())

        blob = cv2.dnn.blobFromImage(colour, 1 / 128, (320, 240), (127, 127, 127), swapRB=True)
        self.ultralight.setInput(blob)
        scores, boxes = self.ultralight.forward(["scores", "boxes"])
        ultralight_best = float(scores[0, :, 1].max())
        return yunet_best, ultralight_best

    def detect(self, picture):
        """{detector: True/False, 'both': True/False} for one grey picture."""
        yunet_best, ultralight_best = self.scores(picture)
        yunet_hit = yunet_best >= self.yunet_score
        ultralight_hit = ultralight_best >= self.ultralight_score
        return {"yunet": bool(yunet_hit), "ultralight": bool(ultralight_hit), "both": bool(yunet_hit and ultralight_hit)}

    def calibrate(self, back_pictures, step=0.05):
        """Thresholds just above the highest score on pictures that certainly show no face (the back of the head)."""
        yunet_scores = []
        ultralight_scores = []
        for picture in back_pictures:
            yunet_best, ultralight_best = self.scores(picture)
            yunet_scores.append(yunet_best)
            ultralight_scores.append(ultralight_best)
        self.yunet_score = round(step * (np.floor(max(yunet_scores) / step) + 1), 2)
        self.ultralight_score = round(min(0.95, step * (np.floor(max(ultralight_scores) / step) + 1)), 2)
        return self.yunet_score, self.ultralight_score


def face_check(head_iso, detectors, views=None):
    """Render the views (default: the 8 VIEWS) and run both detectors. Returns a list of {view, yunet, ultralight, both}."""
    if views is None:
        views = VIEWS
    rows = []
    for yaw, pitch in views:
        picture = render(head_iso, yaw, pitch)
        result = detectors.detect(picture)
        result["view"] = "yaw " + str(yaw) + ", pitch " + str(pitch)
        rows.append(result)
    return rows


# Utility check: a simple tissue segmentation (3 classes inside the brain mask)

def tissue_classes(volume, brain, iterations=20):
    """k-means with 3 classes (CSF, grey matter, white matter on T1) on the brain voxels. 0 = outside the brain."""
    values = volume[brain]
    centres = np.percentile(values, [20, 55, 85]).astype(np.float64)
    for i in range(iterations):
        distances = np.abs(values[:, None] - centres[None, :])
        labels = np.argmin(distances, axis=1)
        for k in range(3):
            if (labels == k).any():
                centres[k] = values[labels == k].mean()
    order = np.argsort(centres)
    rank = np.zeros(3, dtype=int)
    rank[order] = np.arange(3)
    classes = np.zeros(volume.shape, dtype=np.uint8)
    classes[brain] = rank[labels] + 1
    return classes


def dice(a, b):
    total = a.sum() + b.sum()
    if total == 0:
        return 1.0
    return float(2 * (a & b).sum() / total)


# All the checks for one defaced volume

def evaluate(volume, affine, zooms, brain, head, defaced, removed, detectors, check_utility=True):
    """Face check + damage + utility of one defaced volume, compared with the original."""
    voxel_ml = float(np.prod(zooms)) / 1000
    result = {}
    views = face_check(isotropic(head_mask(defaced), zooms), detectors)
    result["views with a face (yunet)"] = sum(1 for v in views if v["yunet"])
    result["views with a face (ultralight)"] = sum(1 for v in views if v["ultralight"])
    result["views with a face (any)"] = sum(1 for v in views if v["yunet"] or v["ultralight"])
    result["views with a face (both)"] = sum(1 for v in views if v["yunet"] and v["ultralight"])
    result["views with a face (both agree)"] = sum(1 for v in views if v["both"])
    result["brain voxels changed"] = int((defaced[brain] != volume[brain]).sum())
    result["removed tissue ml"] = round(float((removed & head).sum()) * voxel_ml, 1)
    if check_utility:
        brain_after = brain_mask(defaced, affine)
        result["skull-strip Dice"] = round(dice(brain, brain_after), 4)
        result["brain volume change %"] = round(100 * (brain_after.sum() - brain.sum()) / brain.sum(), 3)
        classes_before = tissue_classes(volume, brain)
        classes_after = tissue_classes(defaced, brain_after)
        class_dice = []
        for k in [1, 2, 3]:
            class_dice.append(dice(classes_before == k, classes_after == k))
        result["tissue Dice (mean of CSF, GM, WM)"] = round(float(np.mean(class_dice)), 4)
    return result


# DICOM: deface a whole series (head MR/CT)

HEAD_WORDS = ["HEAD", "BRAIN", "SKULL", "NEURO"]


def is_head_series(ds):
    """Head scans are found by Body Part Examined, or by words in the descriptions."""
    text = " ".join([str(ds.get("BodyPartExamined", "")), str(ds.get("StudyDescription", "")),
                     str(ds.get("SeriesDescription", ""))]).upper()
    return any(word in text for word in HEAD_WORDS) and str(ds.get("Modality", "")) in ["MR", "CT"]


def series_volume(file_paths):
    """Stack the slices of one series (sorted by position) -> (pairs [(dataset, path)], volume[x, y, slice], RAS affine)."""
    import pydicom
    pairs = [(pydicom.dcmread(path), path) for path in file_paths]
    first = pairs[0][0]
    row_direction = np.array(first.ImageOrientationPatient[:3], dtype=float)
    column_direction = np.array(first.ImageOrientationPatient[3:], dtype=float)
    normal = np.cross(row_direction, column_direction)
    pairs.sort(key=lambda pair: float(np.dot(normal, np.array(pair[0].ImagePositionPatient, dtype=float))))
    first = pairs[0][0]
    slices = [ds.pixel_array.astype(np.float32).T for ds, path in pairs]      # (column index, row index)
    volume = np.stack(slices, axis=2)
    spacing = [float(v) for v in first.PixelSpacing]                           # [between rows, between columns]
    if len(pairs) > 1:
        step = np.array(pairs[1][0].ImagePositionPatient, dtype=float) - np.array(first.ImagePositionPatient, dtype=float)
    else:
        step = normal * float(first.get("SliceThickness", 1.0))
    affine_lps = np.eye(4)
    affine_lps[:3, 0] = row_direction * spacing[1]
    affine_lps[:3, 1] = column_direction * spacing[0]
    affine_lps[:3, 2] = step
    affine_lps[:3, 3] = np.array(first.ImagePositionPatient, dtype=float)
    lps_to_ras = np.diag([-1, -1, 1, 1])
    return pairs, volume, lps_to_ras @ affine_lps


def series_ras(file_paths):
    """A DICOM series as a RAS volume (the orientation used by all defacing functions).

    Returns (pairs [(dataset, path)], RAS volume, RAS affine, orientation needed to go back to the slice layout).
    """
    pairs, volume, affine = series_volume(file_paths)
    orientation = nib.orientations.io_orientation(affine)
    canonical = nib.orientations.apply_orientation(volume, orientation)
    canonical_affine = affine @ nib.orientations.inv_ornt_aff(orientation, volume.shape)
    return pairs, canonical, canonical_affine, orientation


def deface_dicom_series(file_paths, output_paths, margin_mm=10.0, above_frontal_pole_mm=15.0, below_frontal_pole_mm=None):
    """Deface one head series: only pixels in the face region change; every other value of the files stays the same."""
    from pydicom.uid import ExplicitVRLittleEndian

    output_of = dict(zip(file_paths, output_paths))
    pairs, canonical, canonical_affine, orientation = series_ras(file_paths)
    zooms = np.sqrt((canonical_affine[:3, :3] ** 2).sum(axis=0))
    brain = brain_mask(canonical, canonical_affine)
    defaced, removed = deface_fpdeid(canonical, brain, zooms, margin_mm, above_frontal_pole_mm, below_frontal_pole_mm)
    back = nib.orientations.ornt_transform(nib.orientations.axcodes2ornt(("R", "A", "S")), orientation)
    removed_original = nib.orientations.apply_orientation(removed, back)
    background = min(int(ds.pixel_array.min()) for ds, path in pairs)      # stored value of the background
    changed_slices = 0
    for index, (ds, path) in enumerate(pairs):
        cut = removed_original[:, :, index].T                              # back to (row, column)
        if cut.any():
            pixels = ds.pixel_array.copy()
            pixels[cut] = background
            ds.PixelData = pixels.tobytes()
            ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
            changed_slices = changed_slices + 1
        ds.save_as(output_of[path], enforce_file_format=False)
    return {"slices": len(pairs), "slices changed": changed_slices,
            "removed voxels": int(removed.sum()), "brain voxels": int(brain.sum())}


def write_series(volume, affine, folder):
    """Write a RAS volume as an axial DICOM MR series (one file per slice). Used to test the DICOM path."""
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    os.makedirs(folder, exist_ok=True)
    study = generate_uid()
    series = generate_uid()
    frame = generate_uid()
    spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    stored = np.clip(volume, 0, None).astype(np.uint16)
    files = []
    for k in range(volume.shape[2]):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        path = os.path.join(folder, str(k).zfill(4) + ".dcm")
        ds = FileDataset(path, {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID = meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID = study
        ds.SeriesInstanceUID = series
        ds.FrameOfReferenceUID = frame
        ds.Modality = "MR"
        ds.BodyPartExamined = "HEAD"
        ds.SeriesDescription = "T1 3D head"
        # The volume is RAS (x grows to the right, y to the front, z up); DICOM uses LPS (x to the left, y to the back).
        # Each picture starts at the front-right corner: going along a row goes to the left (+x in LPS), going down a
        # column goes to the back (+y in LPS), so this is a standard axial slice.
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        position_ras = affine @ np.array([volume.shape[0] - 1, volume.shape[1] - 1, k, 1])
        ds.ImagePositionPatient = [float(-position_ras[0]), float(-position_ras[1]), float(position_ras[2])]
        ds.PixelSpacing = [float(spacing[1]), float(spacing[0])]
        ds.SliceThickness = float(spacing[2])
        ds.Rows = volume.shape[1]
        ds.Columns = volume.shape[0]
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        image = stored[::-1, ::-1, k].T                     # rows: front -> back, columns: right -> left
        ds.PixelData = np.ascontiguousarray(image).tobytes()
        ds.save_as(path, enforce_file_format=True)
        files.append(path)
    return files
