"""Build step 4: citation verification.

Checks every citation the model produces against the corpus (a real statute
section or a real opinion ID) and flags anything that is not found. Catches
confabulated citations, not real citations cited for the wrong proposition.
"""
