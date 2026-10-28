"""Comparison method for the ablation study: clean text with a named-entity recogniser (NER) instead of the fingerprint."""

import re

from fpdeid.fingerprint import CONTEXT_PATTERNS, GENERIC_PATTERNS, tidy

ENTITY_TYPES = ["PERSON", "ORG", "GPE", "LOC", "FAC", "DATE", "NORP"]
MODEL = {}


def get_model():
    """Load spaCy once per process (the model cannot be sent between processes)."""
    if "nlp" not in MODEL:
        import spacy
        MODEL["nlp"] = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
        MODEL["cache"] = {}
    return MODEL["nlp"]


class NerCleaner:

    def __init__(self):
        self.patterns = []
        for label, pattern in CONTEXT_PATTERNS + GENERIC_PATTERNS:
            self.patterns.append((re.compile(pattern), label))

    def find(self, text):
        spans = []
        for regex, label in self.patterns:
            for match in regex.finditer(text):
                spans.append((match.start(), match.end(), label, "generic"))

        nlp = get_model()
        cache = MODEL["cache"]
        if text not in cache:
            document = nlp(text)
            entities = []
            for entity in document.ents:
                if entity.label_ in ENTITY_TYPES:
                    entities.append((entity.start_char, entity.end_char, "ner:" + entity.label_, "spaCy"))
            cache[text] = entities
        spans.extend(cache[text])

        # keep the spans that do not overlap an earlier (longer) one
        spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
        kept = []
        last_end = -1
        for span in spans:
            if span[0] >= last_end:
                kept.append(span)
                last_end = span[1]
        return kept

    def clean(self, text):
        spans = self.find(text)
        if len(spans) == 0:
            return text, []
        pieces = []
        last = 0
        for start, end, category, source in spans:
            pieces.append(text[last:start])
            last = end
        pieces.append(text[last:])
        cleaned = tidy("".join(pieces))
        matches = [(category, source) for start, end, category, source in spans]
        return cleaned, matches
