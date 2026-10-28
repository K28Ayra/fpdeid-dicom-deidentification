"""Phase 3: the PHI fingerprint."""

import datetime
import re
from multiprocessing import Pool

import pydicom

# Where identity is known to live ("home" fields)

ID_KEYWORDS = ["PatientID", "OtherPatientIDs", "AccessionNumber", "StudyID", "IssuerOfPatientID",
               "MedicalRecordLocator", "ClinicalTrialSubjectID", "ClinicalTrialSubjectReadingID",
               "AdmissionID", "PatientInsurancePlanCodeSequence"]
ADDRESS_KEYWORDS = ["PatientAddress", "InstitutionAddress", "ReferringPhysicianAddress", "CountryOfResidence",
                    "RegionOfResidence"]
PHONE_KEYWORDS = ["PatientTelephoneNumbers", "ReferringPhysicianTelephoneNumbers"]
INSTITUTION_KEYWORDS = ["InstitutionName", "InstitutionalDepartmentName", "ClinicalTrialSiteName"]
DEVICE_KEYWORDS = ["StationName", "DeviceSerialNumber", "RetrieveAETitle", "PerformedStationAETitle",
                   "ScheduledStationAETitle", "SourceApplicationEntityTitle", "DetectorID", "PlateID",
                   "GantryID", "SecondaryCaptureDeviceID"]

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December"]


def value_list(element):
    if element.VM > 1:
        return [str(v) for v in element.value]
    return [str(element.value)]


PRODUCT_KEYWORDS = ["Manufacturer", "ManufacturerModelName", "SoftwareVersions",
                    "SecondaryCaptureDeviceManufacturer", "SecondaryCaptureDeviceManufacturerModelName",
                    "SecondaryCaptureDeviceSoftwareVersions"]


def product_names(ds):
    """Manufacturer / model / software names of a file (upper case). They are products, not identifiers."""
    names = set()
    for element in ds.iterall():
        if element.keyword in PRODUCT_KEYWORDS and element.value:
            for value in value_list(element):
                names.add(value.strip().upper())
    return names


def collect(ds, found=None):
    """Return a list of (category, source_keyword, raw_value) found in the home fields of one file."""
    if found is None:
        found = []
        products = product_names(ds)
        collect(ds, found)
        # a "device ID" that is just the product name (e.g. the model) does not identify anyone
        kept = []
        for category, source, raw in found:
            if category == "device" and raw.strip().upper() in products:
                continue
            kept.append((category, source, raw))
        return kept
    for element in ds:
        keyword = element.keyword
        if element.VR == "SQ":
            for item in element.value:
                collect(item, found)
            continue
        if element.value is None or str(element.value).strip() == "":
            continue
        if element.tag.is_private:
            continue
        if element.VR == "PN":
            for value in value_list(element):
                found.append(("name", keyword, value))
        elif keyword in ID_KEYWORDS:
            for value in value_list(element):
                found.append(("id", keyword, value))
        elif element.VR == "DA" or keyword == "PatientBirthDate":
            for value in value_list(element):
                found.append(("date", keyword, value))
        elif element.VR == "DT":
            for value in value_list(element):
                found.append(("date", keyword, value[:8]))
        elif keyword in ADDRESS_KEYWORDS:
            for value in value_list(element):
                found.append(("address", keyword, value))
        elif keyword in PHONE_KEYWORDS:
            for value in value_list(element):
                found.append(("phone", keyword, value))
        elif keyword in INSTITUTION_KEYWORDS:
            for value in value_list(element):
                found.append(("institution", keyword, value))
        elif keyword in DEVICE_KEYWORDS:
            for value in value_list(element):
                found.append(("device", keyword, value))
    return found


# Variants: all the ways a value can be written in text

def name_variants(raw):
    """'WILSON^AMY^J' -> full name in several orders + the single parts."""
    parts = []
    for piece in raw.replace("^", " ").replace(",", " ").split():
        if piece != "":
            parts.append(piece)
    if len(parts) == 0:
        return [], []

    full = []
    family = parts[0]
    given = parts[1:]
    if len(given) > 0:
        full.append(" ".join(given) + " " + family)          # Amy J Wilson
        full.append(given[0] + " " + family)                 # Amy Wilson
        full.append(family + " " + " ".join(given))          # Wilson Amy J
        full.append(family + " " + given[0])                 # Wilson Amy
        full.append(given[0][0] + " " + family)              # A Wilson (as "A. Wilson")
    else:
        full.append(family)

    singles = []
    for part in parts:
        if len(part) >= 3:
            singles.append(part)
    return full, singles


def id_variants(raw):
    raw = raw.strip()
    variants = []
    if len(raw) >= 4:
        variants.append(raw)
    without_zeros = raw.lstrip("0")
    if len(without_zeros) >= 4 and without_zeros != raw:
        variants.append(without_zeros)
    digits = re.sub(r"\D", "", raw)
    if len(digits) >= 6 and digits != raw:
        variants.append(digits)
    if len(digits) == 9:
        variants.append(digits[:3] + "-" + digits[3:5] + "-" + digits[5:])   # social-security style
    return variants


def date_variants(raw):
    raw = raw.strip()[:8]
    try:
        date = datetime.datetime.strptime(raw, "%Y%m%d")
    except ValueError:
        return []
    y = date.strftime("%Y")
    yy = date.strftime("%y")
    m = date.strftime("%m")
    d = date.strftime("%d")
    month_name = MONTHS[date.month - 1]
    short_month = month_name[:3]
    variants = [
        y + m + d,
        y + "-" + m + "-" + d, y + "/" + m + "/" + d, y + "." + m + "." + d,
        m + "/" + d + "/" + y, d + "/" + m + "/" + y,
        m + "-" + d + "-" + y, d + "-" + m + "-" + y,
        m + "." + d + "." + y, d + "." + m + "." + y,
        m + "/" + d + "/" + yy, d + "/" + m + "/" + yy,
        str(date.month) + "/" + str(date.day) + "/" + y,
        str(date.day) + "/" + str(date.month) + "/" + y,
        month_name + " " + str(date.day) + ", " + y, short_month + " " + str(date.day) + ", " + y,
        str(date.day) + " " + month_name + " " + y, str(date.day) + " " + short_month + " " + y,
        d + "-" + short_month + "-" + y,
    ]
    return variants


def acronym(raw):
    """'Compton, Holmes and Jones Hospital' -> 'CHAJH';  'Malone-Smith Community Hospital' -> 'MSCH'."""
    words = re.split(r"[\s,\-&/]+", raw)
    letters = ""
    for word in words:
        if word != "" and word[0].isalpha():
            letters = letters + word[0].upper()
    if len(letters) >= 3:
        return letters
    return ""


def address_variants(raw):
    """A real address has a house number or postcode; placeholders like 'UNKNOWN' are not used."""
    variants = []
    if any(ch.isdigit() for ch in raw):
        variants.append(raw.strip())
    for part in raw.split(","):
        part = part.strip()
        if len(part) >= 6 and any(ch.isdigit() for ch in part):
            variants.append(part)
    return variants


def phone_variants(raw):
    digits = re.sub(r"\D", "", raw)
    variants = [raw.strip()]
    if len(digits) == 10:
        variants.append(digits)
        variants.append("(" + digits[:3] + ")" + digits[3:6] + "-" + digits[6:])
        variants.append(digits[:3] + "-" + digits[3:6] + "-" + digits[6:])
    return variants


# Generic PHI patterns (backup for PHI that is not in any home field)

# context patterns run BEFORE the fingerprint, so the context word goes together with the name
CONTEXT_PATTERNS = [
    ("pattern:doctor", r"\b(?:Dr\.|Doctor)\s*[A-Z][a-zA-Z'\-]+"),       # "Dr. Torres" (not "Stern Dr Merced")
    ("pattern:doctor", r"(?<![A-Za-z])DR_[A-Z][A-Za-z]+"),                # "..._DR_ZIMMERMAN"
    ("pattern:id", r"(?<=\bfor )\d{6,}\b|(?<=\bMRN )\d{6,}\b|(?<=\bID )\d{6,}\b|(?<=#)\d{6,}\b"),   # "Ordered for 3363075481"
    ("pattern:initials", r"(?<=\bby )[A-Z]{2,3}\b"),                      # "... by HM" (staff initials)
]

# generic patterns run AFTER the fingerprint, for PHI that is in no home field
GENERIC_PATTERNS = [
    ("pattern:ssn", r"\b\d{3}-\d{2}-\d{4}\b"),
    ("pattern:phone", r"(?:\+?1[\s.\-])?\(?\b\d{3}\)?[\s.\-]?\d{3}[\s.\-]\d{4}(?:\s?x\d{1,5})?(?!\d)"),
    ("pattern:email", r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"),
    ("pattern:date", r"\b(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\b"),
    ("pattern:date", r"\b(?:0?[1-9]|1[0-2])[/.\-](?:0?[1-9]|[12]\d|3[01])[/.\-](?:19|20)\d{2}\b"),
    ("pattern:site", r"\bat\s+[A-Z][A-Z&]{1,7}\s*$"),                    # "... at FMCC" (site code at the end)
    ("pattern:address", r"\b\d{1,6}\s[^:;]{3,60}?,\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?\b"),   # "338 Rosston Rd Prescott, AR 71857"
]


# The fingerprint of one patient

def to_regex(variant):
    """Letters/digits must match exactly; any run of spaces/^/,/./- in between may vary."""
    tokens = re.split(r"[\s\^,._\-]+", variant.strip())
    tokens = [re.escape(t) for t in tokens if t != ""]
    if len(tokens) == 0:
        return ""
    middle = r"[\s\^,._\-]*".join(tokens)
    return r"(?<![A-Za-z0-9])" + middle + r"(?![A-Za-z0-9])"


class Fingerprint:

    def __init__(self, collected, common_words=None, use_variants=True, use_patterns=True, site_entries=None):
        """collected: list of (category, source, raw) from collect(); common_words: words NOT to use alone.

        use_variants=False keeps only the raw values (for the ablation study);
        use_patterns=False switches off the generic PHI patterns (also for the ablation study).
        """
        if common_words is None:
            common_words = set()
        self.entries = []          # (variant, category, source)
        seen = set()

        for category, source, raw in collected:
            variants = []
            if category == "name":
                full, singles = name_variants(raw)
                variants = full
                for single in singles:
                    if single.upper() not in common_words:
                        variants.append(single)
            elif category == "id":
                variants = id_variants(raw)
            elif category == "date":
                variants = date_variants(raw)
            elif category == "address":
                variants = address_variants(raw)
            elif category == "phone":
                variants = phone_variants(raw)
            elif category == "institution":
                variants = [raw.strip()]
                short = acronym(raw)
                if short != "" and short not in common_words:
                    variants.append(short)
            elif category == "device":
                if len(raw.strip()) >= 4 and raw.strip().upper() not in common_words:
                    variants = [raw.strip()]

            if not use_variants:
                variants = [raw.strip()]

            for variant in variants:
                key = variant.upper()
                if key in seen or len(variant.strip()) < 3:
                    continue
                seen.add(key)
                self.entries.append((variant, category, source))

        # site level: full names, addresses, phones and long IDs of OTHER patients (one shared matcher)
        self.site = None
        if site_entries is not None:
            if isinstance(site_entries, SiteMatcher):
                self.site = site_entries
            else:
                self.site = SiteMatcher(site_entries)

        # longest first, so "Amy Wilson" wins over "Wilson"
        self.entries.sort(key=lambda entry: len(entry[0]), reverse=True)
        self.compiled = []
        context = CONTEXT_PATTERNS if use_patterns else []
        generic = GENERIC_PATTERNS if use_patterns else []
        for label, pattern in context:
            self.compiled.append((re.compile(pattern), label, "generic"))
        for variant, category, source in self.entries:
            pattern = to_regex(variant)
            if pattern != "":
                self.compiled.append((re.compile(pattern, re.IGNORECASE), category, source))
        if self.site is not None:
            self.compiled.append((self.site, "site", "site"))
        for label, pattern in generic:
            self.compiled.append((re.compile(pattern), label, "generic"))

    def find(self, text):
        """Return a list of (start, end, category, source) for every match, without overlaps."""
        taken = [False] * len(text)
        spans = []
        for regex, category, source in self.compiled:
            for match in regex.finditer(text):
                start, end = match.start(), match.end()
                if any(taken[start:end]):
                    continue
                for i in range(start, end):
                    taken[i] = True
                if category == "site":
                    found_category, found_source = regex.label(match.group(0))
                    spans.append((start, end, found_category, found_source))
                else:
                    spans.append((start, end, category, source))
        spans.sort()
        return spans

    def clean(self, text):
        """Remove every match from the text. Returns (clean_text, matches)."""
        spans = self.find(text)
        if len(spans) == 0:
            return text, []
        pieces = []
        last = 0
        for start, end, category, source in spans:
            pieces.append(text[last:start])
            last = end
        pieces.append(text[last:])
        cleaned = "".join(pieces)
        cleaned = tidy(cleaned)
        matches = [(category, source) for start, end, category, source in spans]
        return cleaned, matches


def normalise(text):
    return re.sub(r"[\s\^,._\-]+", " ", text).strip().upper()


class SiteMatcher:
    """All site-level entries in ONE regular expression (fast), shared by every patient's fingerprint."""

    def __init__(self, entries):
        entries = sorted(entries, key=lambda entry: len(entry[0]), reverse=True)
        self.labels = {}
        parts = []
        for variant, category, source in entries:
            pattern = to_regex(variant)
            if pattern == "":
                continue
            parts.append(pattern)
            self.labels[normalise(variant)] = (category, "other patient: " + source)
        if len(parts) == 0:
            parts = [r"(?!x)x"]          # matches nothing
        self.regex = re.compile("|".join(parts), re.IGNORECASE)

    def finditer(self, text):
        return self.regex.finditer(text)

    def label(self, matched_text):
        return self.labels.get(normalise(matched_text), ("site", "other patient"))


def tidy(text):
    """Clean up what is left after removing pieces: double spaces, dangling ':' or 'for'."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,;])", r"\1", text)
    text = re.sub(r"(\s*:\s*)+$", "", text)
    text = re.sub(r"^(\s*:\s*)+", "", text)
    text = re.sub(r"\s*(:\s*){2,}", " : ", text)
    text = re.sub(r"\s+(for|at|by|on|of|from)\s*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\(\s*\)", "", text)
    return text.strip(" ,;-^_")


# Build the fingerprints for a whole data set (one pass over the headers)

TEXT_VRS = ["LO", "SH", "ST", "LT", "UT", "UC"]


def scan_file(path):
    """Home values + the set of words used in free text, for one file."""
    ds = pydicom.dcmread(path, stop_before_pixels=True)
    patient = str(ds.get("PatientID", ""))
    collected = collect(ds)
    words = set()
    for element in ds.iterall():
        if element.VR in TEXT_VRS and element.value:
            for word in re.findall(r"[A-Za-z]{3,}", str(element.value)):
                words.add(word.upper())
    return patient, collected, words


def site_level_entries(collected_by_patient):
    """Full names, addresses, phone numbers and long IDs of all patients (never single words)."""
    entries = []
    seen = set()
    for patient, collected in collected_by_patient.items():
        for category, source, raw in collected:
            variants = []
            if category == "name":
                full, singles = name_variants(raw)
                variants = [v for v in full if " " in v and len(v.split()[0]) > 1]
            elif category == "address":
                variants = address_variants(raw)[:1]
            elif category == "phone":
                variants = phone_variants(raw)
            elif category == "id":
                variants = [v for v in id_variants(raw) if len(re.sub(r"\D", "", v)) >= 8]
            for variant in variants:
                key = variant.upper()
                if key not in seen and len(variant) >= 6:
                    seen.add(key)
                    entries.append((variant, category, source))
    return entries


def build_all(paths_list, workers=16, common_share=0.02, use_variants=True, use_patterns=True, use_guard=True,
              use_site=True):
    """Returns {patient_id: Fingerprint}.

    A single word is NOT used on its own (only inside a full name) if it appears in the free text of
    more than `common_share` of all patients for whom it is NOT an identity word: then it is an ordinary word
    (for example a patient called "Brain" should not erase "brain" from "MR BRAIN").
    """
    with Pool(workers) as pool:
        per_file = pool.map(scan_file, paths_list, chunksize=50)

    collected_by_patient = {}
    words_by_patient = {}
    for patient, collected, words in per_file:
        collected_by_patient.setdefault(patient, [])
        collected_by_patient[patient].extend(collected)
        words_by_patient.setdefault(patient, set())
        words_by_patient[patient].update(words)

    # identity words of each patient (single name parts, acronyms, device names)
    identity_words = {}
    for patient, collected in collected_by_patient.items():
        own = set()
        for category, source, raw in collected:
            if category == "name":
                full, singles = name_variants(raw)
                for single in singles:
                    own.add(single.upper())
            if category == "institution":
                own.add(acronym(raw))
            if category == "device":
                own.add(raw.strip().upper())
        identity_words[patient] = own

    # in how many patients' free text does each word appear, without being their own identity word?
    other_use = {}
    for patient, words in words_by_patient.items():
        for word in words:
            if word not in identity_words[patient]:
                other_use[word] = other_use.get(word, 0) + 1
    n_patients = len(words_by_patient)
    limit = max(3, int(common_share * n_patients))
    common_words = set()
    for word, count in other_use.items():
        if count >= limit:
            common_words.add(word)

    if not use_guard:
        common_words = set()

    site_entries = None
    if use_site and use_variants:
        site_entries = SiteMatcher(site_level_entries(collected_by_patient))

    fingerprints = {}
    for patient, collected in collected_by_patient.items():
        fingerprints[patient] = Fingerprint(collected, common_words, use_variants, use_patterns, site_entries)
    return fingerprints, common_words
