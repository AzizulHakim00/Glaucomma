"""V4.5 follow-up: support compact integer mask labels such as 0/1/2."""


def _once(code: str, old: str, new: str, label: str) -> str:
    n = code.count(old)
    if n != 1:
        raise RuntimeError(f"V4.5 low-label patch expected one {label}, found {n}")
    return code.replace(old, new, 1)


def apply_v45_lowlabels(code: str) -> str:
    old_binary = '''    bg = _border_mode(m)\n    out = (np.abs(m.astype(np.int16) - bg) > 3).astype(np.float32)\n    return out, float(out.sum() > 20)\n'''
    new_binary = '''    bg = _border_mode(m)\n    unique_values = np.unique(m)\n    # Label masks may use compact class IDs (0/1 or 0/1/2). For a small\n    # discrete palette, every non-background label is semantic foreground.\n    # For a high-cardinality/compressed image, retain the tolerance guard.\n    if len(unique_values) <= 16:\n        out = (m != bg).astype(np.float32)\n    else:\n        out = (np.abs(m.astype(np.int16) - bg) > 3).astype(np.float32)\n    return out, float(out.sum() > 20)\n'''
    code = _once(code, old_binary, new_binary, "binary compact labels")

    old_candidates = '''    min_region = max(10, int(round(m.size * 0.00002)))\n    candidates = [(v, c) for v, c in levels if abs(v - bg) > 3 and c >= min_region]\n'''
    new_candidates = '''    min_region = max(10, int(round(m.size * 0.00002)))\n    compact_palette = len(levels) <= 16\n    candidates = [\n        (v, c) for v, c in levels\n        if (v != bg if compact_palette else abs(v - bg) > 3) and c >= min_region\n    ]\n'''
    code = _once(code, old_candidates, new_candidates, "combined compact labels")
    return code
