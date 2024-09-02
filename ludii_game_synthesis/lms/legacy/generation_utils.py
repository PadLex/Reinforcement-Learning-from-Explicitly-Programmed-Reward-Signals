'''
These are functions taken from the transformers example generation script, which are mostly unused. Keeping them here
just in case we need them
'''
import inspect
from typing import Tuple

import numpy as np
import torch
from transformers import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length

def sparse_model_config(model_config):
    embedding_size = None
    if hasattr(model_config, "hidden_size"):
        embedding_size = model_config.hidden_size
    elif hasattr(model_config, "n_embed"):
        embedding_size = model_config.n_embed
    elif hasattr(model_config, "n_embd"):
        embedding_size = model_config.n_embd

    num_head = None
    if hasattr(model_config, "num_attention_heads"):
        num_head = model_config.num_attention_heads
    elif hasattr(model_config, "n_head"):
        num_head = model_config.n_head

    if embedding_size is None or num_head is None or num_head == 0:
        raise ValueError("Check the model config")

    num_embedding_size_per_head = int(embedding_size / num_head)
    if hasattr(model_config, "n_layer"):
        num_layer = model_config.n_layer
    elif hasattr(model_config, "num_hidden_layers"):
        num_layer = model_config.num_hidden_layers
    else:
        raise ValueError("Number of hidden layers couldn't be determined from the model config")

    return num_layer, num_head, num_embedding_size_per_head


def generate_past_key_values(model, batch_size, seq_len):
    num_block_layers, num_attention_heads, num_embedding_size_per_head = sparse_model_config(model.config)
    if model.config.model_type == "bloom":
        past_key_values = tuple(
            (
                torch.empty(int(num_attention_heads * batch_size), num_embedding_size_per_head, seq_len)
                .to(model.dtype)
                .to(model.device),
                torch.empty(int(num_attention_heads * batch_size), seq_len, num_embedding_size_per_head)
                .to(model.dtype)
                .to(model.device),
            )
            for _ in range(num_block_layers)
        )
    else:
        past_key_values = tuple(
            (
                torch.empty(batch_size, num_attention_heads, seq_len, num_embedding_size_per_head)
                .to(model.dtype)
                .to(model.device),
                torch.empty(batch_size, num_attention_heads, seq_len, num_embedding_size_per_head)
                .to(model.dtype)
                .to(model.device),
            )
            for _ in range(num_block_layers)
        )
    return past_key_values


def prepare_jit_inputs(inputs, model, tokenizer):
    batch_size = len(inputs)
    dummy_input = tokenizer.batch_encode_plus(inputs, return_tensors="pt")
    dummy_input = dummy_input.to(model.device)
    if model.config.use_cache:
        dummy_input["past_key_values"] = generate_past_key_values(model, batch_size, 1)
    dummy_input["attention_mask"] = torch.cat(
        [
            torch.zeros(dummy_input["attention_mask"].shape[0], 1)
            .to(dummy_input["attention_mask"].dtype)
            .to(model.device),
            dummy_input["attention_mask"],
        ],
        -1,
    )
    return dummy_input


class _ModelFallbackWrapper(GenerationMixin):
    __slots__ = ("_optimized", "_default")

    def __init__(self, optimized, default):
        self._optimized = optimized
        self._default = default

    def __call__(self, *args, **kwargs):
        if kwargs["past_key_values"] is None and self._default.config.use_cache:
            kwargs["past_key_values"] = generate_past_key_values(self._default, kwargs["input_ids"].shape[0], 0)
        kwargs.pop("position_ids", None)
        for k in list(kwargs.keys()):
            if kwargs[k] is None or isinstance(kwargs[k], bool):
                kwargs.pop(k)
        outputs = self._optimized(**kwargs)
        lm_logits = outputs[0]
        past_key_values = outputs[1]
        fixed_output = CausalLMOutputWithPast(
            loss=None,
            logits=lm_logits,
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
        )
        return fixed_output

    def __getattr__(self, item):
        return getattr(self._default, item)

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, use_cache=None, **kwargs
    ):
        return self._default.prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs
        )

    def _reorder_cache(
        self, past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` overwrite_cache if [`~PretrainedModel.beam_search`] or
        [`~PretrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return self._default._reorder_cache(past_key_values, beam_idx)
    
def apply_jit(model, tokenizer):
    jit_input_texts = ["enable jit"]
    jit_inputs = prepare_jit_inputs(jit_input_texts, model, tokenizer)
    torch._C._jit_set_texpr_fuser_enabled(False)
    model.config.return_dict = False
    if hasattr(model, "forward"):
        sig = inspect.signature(model.forward)
    else:
        sig = inspect.signature(model.__call__)
    jit_inputs = tuple(jit_inputs[key] for key in sig.parameters if jit_inputs.get(key, None) is not None)
    traced_model = torch.jit.trace(model, jit_inputs, strict=False)
    traced_model = torch.jit.freeze(traced_model.eval())
    traced_model(*jit_inputs)
    traced_model(*jit_inputs)

    model = _ModelFallbackWrapper(traced_model, model)

    return model, tokenizer

#
# Functions to prepare models' input
#

# Padding text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
# in https://github.com/rusiaaman/XLNet-gen#methodology
# and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e
PREFIX = """In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""

def prepare_ctrl_input(args, _, tokenizer, prompt_text, logger):
    if args.temperature > 0.7:
        logger.info("CTRL typically works better with lower temperatures (and lower top_k).")

    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False)
    if not any(encoded_prompt[0] == x for x in tokenizer.control_codes.values()):
        logger.info("WARNING! You are not starting your generation from a control code so you won't get good results")
    return prompt_text


def prepare_xlm_input(args, model, tokenizer, prompt_text):
    # kwargs = {"language": None, "mask_token_id": None}

    # Set the language
    use_lang_emb = hasattr(model.config, "use_lang_emb") and model.config.use_lang_emb
    if hasattr(model.config, "lang2id") and use_lang_emb:
        available_languages = model.config.lang2id.keys()
        if args.xlm_language in available_languages:
            language = args.xlm_language
        else:
            language = None
            while language not in available_languages:
                language = input("Using XLM. Select language in " + str(list(available_languages)) + " >>> ")

        model.config.lang_id = model.config.lang2id[language]
        # kwargs["language"] = tokenizer.lang2id[language]

    # TODO fix mask_token_id setup when configurations will be synchronized between models and tokenizers
    # XLM masked-language modeling (MLM) models need masked token
    # is_xlm_mlm = "mlm" in args.model_name_or_path
    # if is_xlm_mlm:
    #     kwargs["mask_token_id"] = tokenizer.mask_token_id

    return prompt_text


def prepare_xlnet_input(args, _, tokenizer, prompt_text):
    prefix = args.prefix if args.prefix else args.padding_text if args.padding_text else PREFIX
    prompt_text = prefix + prompt_text
    return prompt_text


def prepare_transfoxl_input(args, _, tokenizer, prompt_text):
    prefix = args.prefix if args.prefix else args.padding_text if args.padding_text else PREFIX
    prompt_text = prefix + prompt_text
    return prompt_text


PREPROCESSING_FUNCTIONS = {
    "ctrl": prepare_ctrl_input,
    "xlm": prepare_xlm_input,
    "xlnet": prepare_xlnet_input,
    "transfo-xl": prepare_transfoxl_input,
}

# Different models need different input formatting and/or extra arguments
def apply_preprocessing(args, model, tokenizer, prompt_text):
    requires_preprocessing = args.model_type in PREPROCESSING_FUNCTIONS.keys()
    if requires_preprocessing:
        prepare_input = PREPROCESSING_FUNCTIONS.get(args.model_type)
        preprocessed_prompt_text = prepare_input(args, model, tokenizer, prompt_text)

        if model.__class__.__name__ in ["TransfoXLLMHeadModel"]:
            tokenizer_kwargs = {"add_space_before_punct_symbol": True}
        else:
            tokenizer_kwargs = {}

        encoded_prompt = tokenizer.encode(
            preprocessed_prompt_text, add_special_tokens=False, return_tensors="pt", **tokenizer_kwargs
        )
    else:
        prefix = args.prefix if args.prefix else args.padding_text
        encoded_prompt = tokenizer.encode(prefix + prompt_text, add_special_tokens=False, return_tensors="pt")

    return encoded_prompt