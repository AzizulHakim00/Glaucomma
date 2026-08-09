"""CPU-only audit patch for RimGraph-DG V4.5.

This mode is intentionally non-training. It permits a CPU Colab runtime,
runs dataset discovery + decoded OD/OC validity audit, persists the audit
artifacts, and exits before any model/backbone construction. RGB duplicate
fingerprinting is skipped because it is unrelated to mask validity.
"""


def _replace_once(code: str, old: str, new: str, label: str) -> str:
    count = code.count(old)
    if count != 1:
        raise RuntimeError(f"V4.5 audit-only expected one {label}, found {count}")
    return code.replace(old, new, 1)


def apply_v45_audit_only(code: str) -> str:
    gpu_guard = '''if IN_COLAB and DEVICE.type != "cuda":
    raise RuntimeError(
        "GPU runtime is required. In Colab choose Runtime > Change runtime type > T4 GPU, then rerun the single cell."
    )
'''
    cpu_ok = '''if IN_COLAB and DEVICE.type != "cuda":
    print("[AUDIT ONLY] CPU Colab runtime accepted; no model training will run.", flush=True)
'''
    code = _replace_once(code, gpu_guard, cpu_ok, "Colab GPU guard")

    fingerprint = '    meta["fingerprint"] = [image_fingerprint(p) for p in meta.image_path]\n'
    fingerprint_skip = '    meta["fingerprint"] = None\n    print("[AUDIT ONLY] Skipping RGB duplicate fingerprinting; not required for mask validity.", flush=True)\n'
    code = _replace_once(code, fingerprint, fingerprint_skip, "RGB fingerprinting")

    marker = 'MASK_VALIDITY = validate_decoded_masks()\n'
    pos = code.find(marker)
    if pos < 0:
        raise RuntimeError("V4.5 audit-only could not find decoded-mask audit completion marker")
    end = pos + len(marker)

    finish = r'''
STORE.save_json({
    "status": "completed",
    "mode": "cpu_mask_audit_only",
    "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "run_name": CFG["run_name"],
    "code_revision": CFG["code_revision"],
    "device": str(DEVICE),
}, "MASK_AUDIT_COMPLETED.json")
for marker_path in [LOCAL_RUN / "RUNNING.json", DRIVE_RUN / "RUNNING.json"]:
    marker_path.unlink(missing_ok=True)
print("\n=== V4.5 CPU MASK AUDIT COMPLETED ===", flush=True)
print(f"Local audit: {LOCAL_RUN}", flush=True)
print(f"Drive audit: {DRIVE_RUN}", flush=True)
print(f"Summary CSV: {DRIVE_RUN / 'mask_validity_by_source.csv'}", flush=True)
print(f"Detailed CSV: {DRIVE_RUN / 'mask_validity_audit.csv'}", flush=True)
print("No backbone/model/training was executed.", flush=True)
'''
    return code[:end] + finish + "\n"
