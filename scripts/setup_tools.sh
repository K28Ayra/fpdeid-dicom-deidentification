#!/usr/bin/env bash
# Downloads the external tools into tools/ (they are not stored in git).
# Run once from the project folder:  bash scripts/setup_tools.sh
set -e
cd "$(dirname "$0")/.."
mkdir -p tools
cd tools

# 1. Official MIDI-B scorer (CBIIT), pinned to the commit used for all results
if [ ! -d MIDI_validation_script ]; then
    git clone https://github.com/CBIIT/MIDI_validation_script.git
fi
git -C MIDI_validation_script checkout -q ccc0939317d5ffd14edda33bc04641490d6f2254

# 2. dciodvfy (DICOM validator from dicom3tools): the Ubuntu/Debian package, unpacked here (no system install)
if [ ! -f dicom3tools_deb/root/usr/bin/dciodvfy ]; then
    mkdir -p dicom3tools_deb
    cd dicom3tools_deb
    apt-get download dicom3tools
    dpkg -x dicom3tools_*.deb root
    cd ..
fi

# 3. DICOM PS3.15 in DocBook XML (only needed to rebuild configs/ps315_*.csv)
if [ ! -f dicom_standard/part15.xml ]; then
    mkdir -p dicom_standard
    curl -L -o dicom_standard/part15.xml https://dicom.nema.org/medical/dicom/current/source/docbook/part15/part15.xml
fi

# 4. Face detectors for the defacing check (Phase 7)
mkdir -p face
if [ ! -f face/face_detection_yunet_2023mar.onnx ]; then
    curl -L -o face/face_detection_yunet_2023mar.onnx \
        https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
fi
if [ ! -f face/version-RFB-320.onnx ]; then
    curl -L -o face/version-RFB-320.onnx \
        https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/raw/master/models/onnx/version-RFB-320.onnx
fi

# 5. SynthStrip (brain mask for defacing): script and weights from the FreeSurfer project
mkdir -p synthstrip
if [ ! -f synthstrip/mri_synthstrip ]; then
    curl -L -o synthstrip/mri_synthstrip \
        https://raw.githubusercontent.com/freesurfer/freesurfer/dev/mri_synthstrip/mri_synthstrip
fi
if [ ! -f synthstrip/synthstrip.1.pt ]; then
    curl -L -o synthstrip/synthstrip.1.pt \
        https://ftp.nmr.mgh.harvard.edu/pub/dist/freesurfer/synthstrip/models/synthstrip.1.pt
fi

# The EasyOCR models (tools/easyocr) and the NLTK data of the scorer (tools/nltk_data) are downloaded automatically
# the first time they are needed.
echo "tools ready in $(pwd)"
