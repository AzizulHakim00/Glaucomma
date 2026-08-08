"""V4.4 Colab runtime guards and visible training progress.

This patch keeps the V4.3 scientific objective unchanged. It adds a real
GPU/full-model preflight, explicit backbone + epoch progress, Drive heartbeat
updates, and preserves the final console log instead of clearing it.
"""


def _once(code: str, old: str, new: str, label: str) -> str:
    count = code.count(old)
    if count != 1:
        raise RuntimeError(f"V4.4 expected one {label}, found {count}")
    return code.replace(old, new, 1)


def _extract(code: str, start: str, end: str, label: str):
    a = code.find(start)
    b = code.find(end, a + 1) if a >= 0 else -1
    if a < 0 or b < 0:
        raise RuntimeError(f"V4.4 could not locate {label}")
    return a, b, code[a:b]


def _replace_in_block(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"V4.4 expected one {label} inside block, found {count}")
    return text.replace(old, new, 1)


def apply_v44_runtime(code: str) -> str:
    code = _once(
        code,
        'write_heartbeat("startup_complete", device=str(DEVICE))\n',
        '''write_heartbeat("startup_complete", device=str(DEVICE))
print("\\n=== RIMGRAPH V4.4 RUNTIME ===", flush=True)
print(f"PyTorch: {torch.__version__}", flush=True)
print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
if DEVICE.type == "cuda":
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"GPU memory: {props.total_memory / (1024**3):.2f} GB", flush=True)
print("==============================\\n", flush=True)
''',
        "startup heartbeat",
    )

    backbone_old = '''            model = timm.create_model(name, pretrained=CFG["pretrained"], features_only=True)
            if CFG.get("gradient_checkpointing", True) and hasattr(model, "set_grad_checkpointing"):
                # Re-entrant checkpointing recomputes activations during backward.
                # In-place ReLU/SiLU operations can mutate saved tensors and cause
                # version-counter failures, so make all supported activations safe.
                for module in model.modules():
                    if hasattr(module, "inplace"):
                        try:
                            module.inplace = False
                        except Exception:
                            pass
                model.set_grad_checkpointing(True)
            return model, name
'''
    backbone_new = '''            print(f"[BACKBONE] loading {name} pretrained={CFG['pretrained']} ...", flush=True)
            model = timm.create_model(name, pretrained=CFG["pretrained"], features_only=True)
            if CFG.get("gradient_checkpointing", True) and hasattr(model, "set_grad_checkpointing"):
                for module in model.modules():
                    if hasattr(module, "inplace"):
                        try:
                            module.inplace = False
                        except Exception:
                            pass
                model.set_grad_checkpointing(True)
            print(f"[BACKBONE] ready: {name}", flush=True)
            return model, name
'''
    code = _once(code, backbone_old, backbone_new, "backbone creation")

    a, b, fn = _extract(code, 'def run_baseline_epoch(', '\n\ndef run_full_epoch', "baseline epoch")
    fn = _replace_in_block(
        fn,
        '    all_y, all_p, losses = [], [], []\n',
        '    all_y, all_p, losses = [], [], []\n    if train:\n        print(f"[BASELINE] epoch {epoch}/{CFG[\'baseline_epochs\']} start | batches={len(loader)}", flush=True)\n        write_heartbeat("baseline_epoch_start", epoch=int(epoch), batches=int(len(loader)))\n',
        "baseline epoch start",
    )
    fn = _replace_in_block(
        fn,
        '        batch = batch_to_device(batch)\n',
        '        batch = batch_to_device(batch)\n        if train and (bi == 0 or (bi + 1) % max(1, len(loader) // 4) == 0):\n            print(f"[BASELINE] epoch {epoch} batch {bi + 1}/{len(loader)}", flush=True)\n',
        "baseline batch progress",
    )
    fn = _replace_in_block(
        fn,
        '    metrics["loss"] = float(np.mean(losses))\n    return metrics\n',
        '    metrics["loss"] = float(np.mean(losses))\n    if train:\n        print(f"[BASELINE] epoch {epoch} done | loss={metrics[\'loss\']:.4f} auroc={metrics[\'auroc\']:.4f} auprc={metrics[\'auprc\']:.4f}", flush=True)\n        write_heartbeat("baseline_epoch_complete", epoch=int(epoch), loss=float(metrics["loss"]), auroc=float(np.nan_to_num(metrics["auroc"])), auprc=float(np.nan_to_num(metrics["auprc"])))\n    return metrics\n',
        "baseline epoch completion",
    )
    code = code[:a] + fn + code[b:]

    a, b, fn = _extract(code, 'def run_full_epoch(', '\n\n@torch.no_grad()\ndef predict_baseline', "full epoch")
    fn = _replace_in_block(
        fn,
        '    stage = full_stage(epoch)\n',
        '    stage = full_stage(epoch)\n    if train:\n        print(f"[RIMGRAPH] epoch {epoch}/{CFG[\'full_epochs\']} start | stage={stage} | batches={len(loader)}", flush=True)\n        write_heartbeat("rimgraph_epoch_start", epoch=int(epoch), model_stage=str(stage), batches=int(len(loader)))\n',
        "full epoch start",
    )
    fn = _replace_in_block(
        fn,
        '        batch = batch_to_device(batch)\n',
        '        batch = batch_to_device(batch)\n        if train and (bi == 0 or (bi + 1) % max(1, len(loader) // 4) == 0):\n            print(f"[RIMGRAPH] epoch {epoch} stage={stage} batch {bi + 1}/{len(loader)}", flush=True)\n',
        "full batch progress",
    )
    marker = '    metrics["prototype_active_batch_ratio"] = float(prototype_active_batches / full_stage_batches) if full_stage_batches else np.nan\n'
    fn = _replace_in_block(
        fn,
        marker,
        marker + '''    if train:
        proto = metrics["prototype_active_batch_ratio"]
        proto_text = "n/a" if np.isnan(proto) else f"{proto:.3f}"
        print(f"[RIMGRAPH] epoch {epoch} done | stage={stage} loss={metrics['loss']:.4f} auroc={metrics['auroc']:.4f} auprc={metrics['auprc']:.4f} disc_dice={metrics['dice_disc']:.4f} cup_dice={metrics['dice_cup']:.4f} proto_active={proto_text}", flush=True)
        write_heartbeat("rimgraph_epoch_complete", epoch=int(epoch), model_stage=str(stage), loss=float(metrics["loss"]), auroc=float(np.nan_to_num(metrics["auroc"])), auprc=float(np.nan_to_num(metrics["auprc"])), prototype_active_batch_ratio=None if np.isnan(proto) else float(proto))
''',
        "full epoch completion",
    )
    code = code[:a] + fn + code[b:]

    preflight = '''def _v44_gpu_model_preflight():
    print("[PREFLIGHT] constructing full RimGraph model ...", flush=True)
    write_heartbeat("gpu_model_preflight_start")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model = RimGraphV4(num_domains=2).to(DEVICE)
    model.train()
    x = torch.randn(2, 3, int(CFG["image_size"]), int(CFG["image_size"]), device=DEVICE, requires_grad=True)
    with torch.autocast(device_type=DEVICE.type, enabled=CFG["mixed_precision"] and DEVICE.type == "cuda"):
        out = model(x, stage="full", grl_lambda=0.1)
        probe_loss = out["logit"].float().mean() + 0.01 * out["seg"].float().mean() + 0.01 * out["domain"].float().mean()
    probe_loss.backward()
    if not torch.isfinite(probe_loss.detach()):
        raise RuntimeError("V4.4 GPU model preflight produced a non-finite loss")
    peak = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    print(f"[PREFLIGHT] full-model forward/backward PASSED | peak allocated={peak:.2f} GB", flush=True)
    write_heartbeat("gpu_model_preflight_passed", peak_allocated_gb=float(peak))
    model.to("cpu")
    del model, x, out, probe_loss
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


_v44_gpu_model_preflight()

ALL_METRICS, ALL_PREDS = [], []
'''
    code = _once(code, 'ALL_METRICS, ALL_PREDS = [], []\n', preflight, "GPU model preflight")

    code = _once(
        code,
        'clear_output(wait=True)\ndisplay(Markdown("# ✅ RimGraph-DG V4.1 run completed"))\n',
        'print("\\n=== RIMGRAPH V4.4 RUN COMPLETED ===", flush=True)\ndisplay(Markdown("# ✅ RimGraph-DG V4.4 run completed"))\n',
        "completion display",
    )

    return code
