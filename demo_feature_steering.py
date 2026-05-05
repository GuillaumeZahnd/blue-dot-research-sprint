import torch
import torch.nn.functional as F
from contextlib import contextmanager

from llm import select_llm
from queries import Queries
from analyze_features import resolve_hook_point
from parameters import Parameters


class FeatureIntervention:
    def __init__(self, model, sae, tokenizer, device="cuda"):
        self.model = model
        self.sae = sae
        self.tokenizer = tokenizer
        self.device = device
        self.hook_point, self.prepend_bos, self.context_size = resolve_hook_point(sae=sae)

    @contextmanager
    def apply_intervention(self, feature_indices, mode: str, intensity: float):
        w_dec         = self.sae.W_dec[feature_indices].to(self.device).to(self.model.cfg.dtype)
        W_enc_sub     = self.sae.W_enc[:, feature_indices].to(self.device).to(self.model.cfg.dtype)
        b_enc_sub     = self.sae.b_enc[feature_indices].to(self.device).to(self.model.cfg.dtype)
        b_dec         = self.sae.b_dec.to(self.device).to(self.model.cfg.dtype)
        threshold_sub = self.sae.threshold[feature_indices].to(self.device).to(self.model.cfg.dtype)

        def hook_fn(activations, hook):
            with torch.no_grad():
                pre_acts = torch.einsum("bsd,df->bsf", activations - b_dec, W_enc_sub) + b_enc_sub
                feature_acts = pre_acts * (pre_acts > threshold_sub)  # JumpReLU

            if mode == "zero":
                correction = torch.einsum("bsf,fd->bsd", feature_acts, w_dec)
                return activations - correction
            elif mode == "amplify":
                correction = torch.einsum("bsf,fd->bsd", feature_acts * (intensity - 1.0), w_dec)
                return activations + correction
            elif mode == "baseline":
                return activations

        try:
            self.model.add_hook(self.hook_point, hook_fn)
            yield
        finally:
            self.model.reset_hooks()


    def run_test(self, prompt, feature_indices, mode="amplify", intensity: float=None, max_tokens: int=128):
        """Helper to run a prompt with the intervention active."""
        self.model.eval()

        print(f"\n--- Running Mode: {mode} (Features: {feature_indices}) ---")

        with self.apply_intervention(feature_indices, mode, intensity):
            output_text = self.model.generate(
                prompt,
                max_new_tokens=max_tokens,
                stop_at_eos=True,
                eos_token_id=self.tokenizer.eos_token_id,
                prepend_bos=self.prepend_bos,
                verbose=False
            )

        return output_text


if __name__ == "__main__":

    model_nickname = Parameters.MODEL_NICKNAME
    layer = Parameters.TARGET_LAYER_INDEX
    dtype = Parameters.DTYPE

    model, tokenizer, sae = select_llm(model_nickname=model_nickname, layer=layer, dtype=dtype)
    editor = FeatureIntervention(model, sae, tokenizer)

    # Based on the outcome of "analyze_features.py"
    indices = [12227, 20768, 22358, 6953, 296, 4420, 19243, 6886, 26576, 25406, 28487, 23835, 19544, 23172, 6308, 3606]

    intensity_scaling = 20.0

    query_harmless = Queries.HARMLESS
    query_harmful = Queries.HARMFUL
    query_false_positive = Queries.FALSE_POSITIVE

    print("="*64 + "\n" + query_harmless + "\n" + "="*64)
    print(editor.run_test(query_harmless, indices, mode="baseline"))
    print(editor.run_test(query_harmless, indices, mode="amplify", intensity=intensity_scaling))
    print(editor.run_test(query_harmless, indices, mode="zero"))

    print("="*64 + "\n" + query_harmful + "\n" + "="*64)
    print(editor.run_test(query_harmful, indices, mode="baseline"))
    print(editor.run_test(query_harmful, indices, mode="amplify", intensity=intensity_scaling))
    print(editor.run_test(query_harmful, indices, mode="zero"))

    print("="*64 + "\n" + query_false_positive + "\n" + "="*64)
    print(editor.run_test(query_false_positive, indices, mode="baseline"))
    print(editor.run_test(query_false_positive, indices, mode="amplify", intensity=intensity_scaling))
    print(editor.run_test(query_false_positive, indices, mode="zero"))

