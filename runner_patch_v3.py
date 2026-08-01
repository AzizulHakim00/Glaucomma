"""Additional runtime fixes applied after runner_patch_v2."""


def _replace_once(code: str, old: str, new: str, name: str) -> str:
    count = code.count(old)
    if count != 1:
        raise RuntimeError(f"Patch {name!r} expected exactly one match, found {count}.")
    return code.replace(old, new, 1)


def apply_patches_v3(code: str) -> str:
    # Validation/inference must not construct autograd graphs. This substantially
    # reduces GPU memory usage and prevents avoidable Colab OOM failures.
    code = _replace_once(
        code,
        '        with torch.autocast(device_type=DEVICE.type,enabled=CFG["mixed_precision"] and DEVICE.type=="cuda"):',
        '        with torch.set_grad_enabled(train), torch.autocast(device_type=DEVICE.type,enabled=CFG["mixed_precision"] and DEVICE.type=="cuda"):',
        "disable-validation-gradients",
    )

    # Anatomical consistency is meaningful only when a verified segmentation
    # annotation exists. Unannotated classification samples must not receive a
    # fabricated structural target.
    code = _replace_once(
        code,
        '            l_cons=F.smooth_l1_loss(o["vcdr_reg"],o["vcdr_mask"].detach())',
        '            l_cons_raw=F.smooth_l1_loss(o["vcdr_reg"],o["vcdr_mask"].detach(),reduction="none")\n            l_cons=(l_cons_raw*b["mask_valid"]).sum()/b["mask_valid"].sum().clamp_min(1)',
        "mask-anatomical-consistency",
    )

    # Resume only checkpoints produced by the exact same runner revision.
    old = '''    if CFG["resume"] and last:
        ck=torch.load(last,map_location=DEVICE,weights_only=False);model.load_state_dict(ck["model"]);optimizer.load_state_dict(ck["optimizer"]);scheduler.load_state_dict(ck["scheduler"]);history=ck["history"];best_score=ck["best_score"];best_epoch=ck["best_epoch"];start_epoch=ck["epoch"]+1'''
    new = '''    if CFG["resume"] and last:
        ck=torch.load(last,map_location=DEVICE,weights_only=False)
        saved_revision=ck.get("cfg",{}).get("code_revision")
        current_revision=CFG.get("code_revision")
        if saved_revision == current_revision:
            model.load_state_dict(ck["model"])
            optimizer.load_state_dict(ck["optimizer"])
            scheduler.load_state_dict(ck["scheduler"])
            history=ck["history"]
            best_score=ck["best_score"]
            best_epoch=ck["best_epoch"]
            start_epoch=ck["epoch"]+1
            display(Markdown(f"Resuming `{target_source}` from epoch **{start_epoch}**."))
        else:
            display(Markdown(
                f"**Ignoring an incompatible old checkpoint for {target_source}.** "
                f"Saved revision: `{saved_revision}`; current revision: `{current_revision}`."
            ))'''
    code = _replace_once(code, old, new, "revision-safe-partial-resume")

    # A completed fold is reusable only when both run mode and code revision match.
    old = '''        completed_mode = bool(saved.get("fast_dev_run", False))
        if completed_mode == bool(CFG["fast_dev_run"]):'''
    new = '''        completed_mode = bool(saved.get("fast_dev_run", False))
        completed_revision = saved.get("code_revision")
        if completed_mode == bool(CFG["fast_dev_run"]) and completed_revision == CFG.get("code_revision"):'''
    code = _replace_once(code, old, new, "revision-safe-completed-resume")

    old = '''    STORE.save_json({"metrics":metrics,"fast_dev_run":bool(CFG["fast_dev_run"]),"completed_at":time.strftime("%Y-%m-%d %H:%M:%S")},f"{fold_rel}/COMPLETED.json")'''
    new = '''    STORE.save_json({"metrics":metrics,"fast_dev_run":bool(CFG["fast_dev_run"]),"code_revision":CFG.get("code_revision"),"completed_at":time.strftime("%Y-%m-%d %H:%M:%S")},f"{fold_rel}/COMPLETED.json")'''
    code = _replace_once(code, old, new, "write-completed-revision")

    # Guard against an empty or one-class validation prediction table before
    # calibration/threshold fitting, producing a clear actionable error.
    old = '''    val_pred,_=predict(model,val_loader)
    T_standard=fit_temperature(val_pred.label.values,val_pred.prob_raw.values)'''
    new = '''    val_pred,_=predict(model,val_loader)
    if val_pred.empty or val_pred["label"].nunique() < 2:
        raise RuntimeError(
            f"Validation split for held-out {target_source} does not contain both classes. "
            "Increase the labelled source data or val_fraction before calibration."
        )
    T_standard=fit_temperature(val_pred.label.values,val_pred.prob_raw.values)'''
    code = _replace_once(code, old, new, "calibration-class-guard")

    return code
