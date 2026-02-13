import os


def patch_qwen_weights_vllm():
    """
    Patch weight names of qwen multimodal models consistently with transformers==4.52
    See https://github.com/vllm-project/vllm/pull/19054
    """
    try:
        import vllm
        from vllm.model_executor.models.utils import WeightsMapper

        registry = vllm.model_executor.models.ModelRegistry.models

        def _apply(model_key: str):
            try:
                registry[model_key].load_model_cls().hf_to_vllm_mapper = WeightsMapper(
                    orig_to_new_prefix={
                        "model.language_model.": "language_model.model.",
                        "model.visual.": "visual.",
                        "lm_head.": "language_model.lm_head.",
                        "model.": "language_model.model.",
                    }
                )
                return True
            except Exception:
                return False

        ok1 = _apply("Qwen2_5_VLForConditionalGeneration")
        ok2 = _apply("Qwen2VLForConditionalGeneration")
        if ok1 or ok2:
            print("### Patch to vllm qwen modelling applied successfully.", flush=True)
    except Exception:
        pass


if os.environ.get("DISABLE_QWEN_VLLM_PATCH") != "1":
    patch_qwen_weights_vllm()


