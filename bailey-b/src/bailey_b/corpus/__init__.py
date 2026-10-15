"""Build step 1: corpus download and chunking.

Statutes first (``corpus.statutes``): per-chapter pages from flsenate.gov,
chunked one section per chunk. Then CourtListener (federal), Second DCA and Florida Supreme Court
opinions, and the Caselaw Access Project. LexisNexis is NOT a source.
Raw downloads go under ``<data_dir>/raw/``, chunk JSONL under ``<data_dir>/chunks/``.
"""
