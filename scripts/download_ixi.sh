#!/usr/bin/env bash
# Downloads 30 T1 head MRI scans of the IXI data set (CC BY-SA 3.0) into data/ixi/t1 (Phase 7).
# IXI only offers one 4.8 GB archive of all T1 scans. It is a plain tar file, so the first 240 MB already contain
# about 30 complete scans: only that part is downloaded (16 parallel range requests), and broken last files are removed.
set -e
cd "$(dirname "$0")/.."
URL=https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T1.tar
OUT=data/ixi
mkdir -p $OUT/parts $OUT/t1
CHUNK=15000000
for i in $(seq 0 15); do
    START=$((i * CHUNK))
    END=$((START + CHUNK - 1))
    curl -sL --retry 5 -r $START-$END -o $OUT/parts/part_$(printf %02d $i) $URL &
done
wait
cat $OUT/parts/part_* > $OUT/first.tar
tar -xf $OUT/first.tar -C $OUT/t1 2>/dev/null || true
for f in $OUT/t1/*.nii.gz; do
    gzip -t "$f" 2>/dev/null || rm "$f"
done
rm -rf $OUT/parts $OUT/first.tar
echo "$(ls $OUT/t1 | wc -l) scans in $OUT/t1"
