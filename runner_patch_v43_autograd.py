"""Autograd safety patch for gradient-checkpointed timm backbones."""


def apply_v43_autograd(code: str) -> str:
    old = '''            model = timm.create_model(name, pretrained=CFG["pretrained"], features_only=True)
            if CFG.get("gradient_checkpointing", True) and hasattr(model, "set_grad_checkpointing"):
                model.set_grad_checkpointing(True)
            return model, name
'''
    new = '''            model = timm.create_model(name, pretrained=CFG["pretrained"], features_only=True)
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
    count = code.count(old)
    if count != 1:
        raise RuntimeError(f"V4.3 autograd patch expected one backbone block, found {count}")
    return code.replace(old, new, 1)
