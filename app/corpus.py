def select_doc_ids(sources, source=None, max_docs=None):
    """Return dataset row indices to index.

    sources: list of per-row source labels (e.g. the `dataset_name` column).
    source:  keep only rows whose label equals this (None = keep all).
    max_docs: cap on how many rows to keep (None = no cap).
    Row indices are kept as-is so doc_id always refers to the original row.
    """
    ids = [i for i, s in enumerate(sources) if source is None or s == source]
    return ids[:max_docs] if max_docs else ids
