"""FP-DeID: fingerprint-guided, closed-loop DICOM de-identification."""

import warnings

# The synthetic MIDI-B input UIDs contain parts with leading zeros (e.g. ".070."),
# which pydicom warns about on every file. The warning is harmless, so we hide it.
warnings.filterwarnings("ignore", message="Invalid value for VR UI")
# Some synthetic files have a malformed Specific Character Set ("ISO_IR" "100" as two values).
# pydicom falls back to the default encoding, which is the right behaviour, so hide that warning too.
warnings.filterwarnings("ignore", message="Unknown encoding")
