import argparse
import json
import os
from pathlib import Path
import re
import sys

from datasets import load_dataset
import inflect
from transformers import AutoModelForCausalLM, GenerationConfig, set_seed
import urwid

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from java_api import FastTrace, Autocomplete
from ludii_datasets import _format_fitm, LudiiDataset, _mask_names
from utils import load_model_and_tokenizer, format_single_line_game, format_multi_line_game

class GameDisplay(urwid.ListBox):
    def __init__(self, model, tokenizer, config, dataset):
        super().__init__(urwid.SimpleFocusListWalker([]))
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.dataset = dataset

        self.mode = "game_selection"

        self.eval = FastTrace()
        self.autocomplete = Autocomplete()

        self._show_game_selection()

    def _show_game_selection(self):
        # names = set()
        # for game in self.dataset.val['text']:
        #     name = re.findall(r'"(.*?)"', game)[0].replace("<SUF>", "").strip()
        #     names.add(name)

        names = self.dataset['name']

        self.game_names = list(sorted(names))

        body = []
        for name in self.game_names:
            button = urwid.Button(name)
            body.append(urwid.AttrMap(button, None, focus_map='reversed'))

        self.body = body


    def _load_game(self, game: str):
        self.raw_game = game

        self.raw_game = _mask_names(self.raw_game, inflect.engine())

        self.formatted_game = format_single_line_game(self.raw_game)
        self.game_lines = self.formatted_game.split("\n")
        self.selected_lines = []

        body = [urwid.Button("[DEBUG]"), urwid.Button("[DEBUG]")]
        for line in self.game_lines:
            button = urwid.Button(line)
            body.append(urwid.AttrMap(button, None, focus_map='reversed'))

        self.body = body

    def _format_output(self, output_token_ids: str):
        '''
        Convert token ids from FITM output into a string
        '''
        output = self.tokenizer.decode(output_token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True)
        output = output.split("<MID> ")[1]
        output = self.tokenizer.decode(self.tokenizer.encode(output), skip_special_tokens=True)

        return output

    def keypress(self, size, key):
        key_str = str(key)
        super().keypress(size, key)

        if key_str == 'esc':
            if self.mode == "edit":
                if len(self.selected_lines) > 0:
                    self._reset_selected_lines()
                else:
                    self.mode = "game_selection"
                    self._show_game_selection()
            else:
                raise urwid.ExitMainLoop()
        
        if key_str == 'enter':
            if self.mode == "game_selection":
                path = Path("./ludii_data/games/expanded").rglob(self.game_names[self.focus_position] + ".lud").__next__().as_posix()
                self._load_game(open(path, "r").read())
                self.mode = "edit"
        
            elif self.mode == "edit":
                if len(self.selected_lines) > 0:
                    prefix = format_multi_line_game("\n".join([self.game_lines[idx] for idx in range(self.selected_lines[0])]))
                    suffix = format_multi_line_game("\n".join([self.game_lines[idx] for idx in range(self.selected_lines[-1] + 1, len(self.game_lines))]))
                    target = format_multi_line_game("\n".join([self.game_lines[idx] for idx in self.selected_lines]))

                    prompt, _ = _format_fitm((prefix, target, suffix, None)).split("<MID>")
                    
                    # possibe_continuations = self._get_possible_continuations(prefix)
                    # continuation = random.choice(possibe_continuations) + " "
                    continuation = ""
                    prompt = prompt + "<MID>" + continuation

                    inputs = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
                    generation_outputs = self.model.generate(
                        input_ids=inputs,
                        generation_config=self.config,
                        max_new_tokens=1024
                    )

                    outputs = [self._format_output(output) for output in generation_outputs]

                    main_output = outputs[0]

                    # Fitness eval before
                    # score_before = [self.eval.evaluate(self.raw_game) for _ in range(1)]

                    new_game = (prefix + main_output + suffix).strip()
                    self._load_game(new_game)

                    # Fitness eval before
                    # score_after = [self.eval.evaluate(self.raw_game) for _ in range(1)]

                    # For debugging               
                    self.body[0].base_widget.set_label(f"[DEBUG] Selected section:         {target}")
                    for i, output in enumerate(outputs):
                        self.body[i + 1].base_widget.set_label(f"[DEBUG] Generated continuation {i}: {output}")

                    # pos = len(outputs) + 1
                    # self.body[pos].base_widget.set_label(f"[DEBUG] Fitness before: {score_before}")
                    # self.body[pos + 1].base_widget.set_label(f"[DEBUG] Fitness after:  {score_after}")

                    self._reset_selected_lines()
                            
                else:
                    self.selected_lines = self._get_selected_lines(self.focus_position-2) # Account for the two debug lines at the top
                    for line_idx in self.selected_lines:
                        line_idx += 2 # Account for the two debug lines at the top
                        self.body[line_idx] = urwid.AttrMap(self.body[line_idx].base_widget, 'selected', 'selected_focus')

    def _get_possible_continuations(self, partial_game: str):
        '''
        Use the autocomplete endpoint to get the possible continuations of the partial game
        
        WIP
        '''

        completions = self.autocomplete.next_tokens(partial_game)
        return completions

    def _reset_selected_lines(self):
        for idx, line in enumerate(self.body):
            self.body[idx] = urwid.AttrMap(line.base_widget, None, 'reversed')

        self.selected_lines = []

    def _get_selected_lines(self, idx: int):
        '''
        Returns the indices of every line that is contained within the line at the current focus position
        '''
        contained_lines = []

        if not self.game_lines[idx].strip().startswith("("):
            return contained_lines
        
        depth = 0
        for addl_idx, line in enumerate(self.game_lines[idx:]):
            contained_lines.append(addl_idx + idx)

            depth += line.count("(")
            depth -= line.count(")")

            if depth == 0:
                break

        return contained_lines



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_dir', type=str, default='./logs/code-llama-13b-fitm-mask', help="Directory to load the model from")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_return_sequences', type=int, default=1)

    # Hyperparameters
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--no_sample', action='store_true')
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--top_p', type=float, default=1)
    parser.add_argument('--typical_p', type=float, default=1)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--num_beam_groups', type=int, default=1)
    parser.add_argument('--repetition_penalty', type=float, default=1)
    parser.add_argument('--penalty_alpha', type=float, default=1)
    parser.add_argument('--diversity_penality', type=float, default=0.0)

    args = parser.parse_args()

    set_seed(args.seed)

    # model, tokenizer = load_model_and_tokenizer(args.load_dir)
    model_name = os.path.basename(args.load_dir)
    model = AutoModelForCausalLM.from_pretrained(f"LudiiLMs/{model_name}")
    tokenizer = load_model_and_tokenizer(args.load_dir, only_tokenizer=True)

    run_args = argparse.Namespace() 
    run_args.__dict__ = json.load(open(os.path.join(args.load_dir, "run_args.json"), "r"))

    # dataset = load_dataset("LudiiLMs/" + model_name + "-base-data", split="train")
    dataset = load_dataset("LudiiLMs/" + "code-llama-13b-fitm-mask" + "-base-data", split="train")


    config = GenerationConfig(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        typical_p=args.typical_p,
        num_beams=args.num_beams,
        num_beam_groups=args.num_beam_groups,
        repetition_penalty=args.repetition_penalty,
        penalty_alpha=args.penalty_alpha,
        diversity_penalty=args.diversity_penality,

        do_sample=not args.no_sample,
        num_return_sequences=args.num_return_sequences,
        pad_token_id=tokenizer.eos_token_id,
        renormalize_logits=True,
    )


    test_game = open("./ludii_data/games/expanded/board/space/line/Fettas.lud", "r").read()

    palette = [("reversed", "standout", ""),
               ("normal", "", ""),
               ("selected", "", "dark green"),
               ("selected_focus", "", "dark blue")]

    display = GameDisplay(model, tokenizer, config, dataset)
    # display._load_game(test_game)
    urwid.MainLoop(display, palette).run()
