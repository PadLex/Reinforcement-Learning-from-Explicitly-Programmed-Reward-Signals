import re
import time
import torch
from transformers import LogitsProcessor

from java_api import Autocomplete, Compile, PRIMITIVES

class PICARDGrammarConstraint(LogitsProcessor):
    '''
    A logits processor based off of PICARD that ensures generated games are
    always compilable

    TODO: we can specify that top-k has already been applied, which is probably a good
          call to make sure we don't re-do the work
    '''
    def __init__(self,
                 tokenizer,
                 backoff_k=25,
                 verbose=False):
        
        self.tokenizer = tokenizer
        self.compiler = Compile()
        self.autocomplete = Autocomplete()
        self.backoff_k = backoff_k
        self.verbose = verbose

        self.autocomplete_calls = []

    def _autocomplete(self, game, overwrite_cache=True):
        '''
        Helper function for autocompleting games that will re-try if the compiler fails
        '''
        done = False
        while not done:
            try:
                self.autocomplete_calls.append(game)
                next_tokens = self.autocomplete.next_tokens(game, overwrite_cache=overwrite_cache)
                done = True

            except SyntaxError:
                next_tokens = []
                done = True

            except BrokenPipeError:
                self.autocomplete = Autocomplete()

        return next_tokens
    
    def _is_valid_token(self, token_idx):
        '''
        Fitler out LM tokens that begin with backslashes or carats, that aren't the EOS token
        '''

        if repr(self.tokenizer.decode(token_idx)).startswith("'\\"):
            return False
        
        elif repr(self.tokenizer.decode(token_idx)).startswith("'<") and token_idx != self.tokenizer.eos_token_id:
            return False

        return True

    def _get_k(self, valid_ludii_tokens):
        '''
        Convert a list of valid ludii tokens into a value for k -- the restriction on the most likely
        LM tokens. This is a heuristic method that's not guaranteed to be correct, but should overestimate
        the number of valid LM tokens
        '''
        
        # Check if the ludii tokens contains any primitives
        if not any([primitive in ludii_token for primitive in PRIMITIVES for ludii_token in valid_ludii_tokens]):
            return len(valid_ludii_tokens)
        
        elif "CONTINUED_STRING?" in valid_ludii_tokens:
            return self.backoff_k
        
        elif "NEW_STRING?" in valid_ludii_tokens:
            return 5
        
        elif "NEW_BOOLEAN?" in valid_ludii_tokens or "CONTINUED_BOOLEAN?" in valid_ludii_tokens:
            return 5
        
        elif "NEW_INT?" in valid_ludii_tokens or "CONTINUED_INT?" in valid_ludii_tokens:
            return 15
        
        elif "NEW_DIM?" in valid_ludii_tokens or "CONTINUED_DIM?" in valid_ludii_tokens:
            return 15
        
        elif "NEW_FLOAT?" in valid_ludii_tokens or "CONTINUED_FLOAT?" in valid_ludii_tokens:
            return 15
        
        else:
            return self.backoff_k

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:

        # Iterate over all inputs and scores
        for batch_idx, token_ids in enumerate(input_ids):
            original_scores = scores[batch_idx].clone()

            initial_game = self.tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            valid_ludii_tokens = self._autocomplete(initial_game, overwrite_cache=True)

            if self.verbose:
                print(f"\nCurrent game: '{initial_game}'")
                print(f"Most recent generated token: {repr(self.tokenizer.decode(token_ids[-1]))}")
                print(f"Valid Ludii tokens: {valid_ludii_tokens}")

            if self.backoff_k is not None:
           
                # Restrict to only the K most likely tokens, as heuristically determined by the autocomplete
                top_k = min(self._get_k(valid_ludii_tokens), self.backoff_k)

                try:
                    indices_to_remove = scores[batch_idx] < torch.topk(scores[batch_idx], top_k)[0][..., -1, None]
                    scores[batch_idx] = scores[batch_idx].masked_fill(indices_to_remove, -torch.inf)
                
                # Emergency backoff -- if we can't even get the top-k, then we just set the EOS token to 0
                except IndexError:
                    if self.verbose:
                        print(f"Emergency top-k backoff triggered for game: {initial_game}")
                    
                    scores[batch_idx] = torch.full_like(scores[batch_idx], -torch.inf)
                    scores[batch_idx][self.tokenizer.eos_token_id] = 0
                    continue

            
            # Extract all the indices where the score hasn't already been set to -inf
            next_token_indices = torch.argwhere(scores[batch_idx] > -torch.inf)

            # Check whether the continuations are compilable
            for token_idx in next_token_indices:
                continuation = torch.cat((token_ids, token_idx))
                partial_game = self.tokenizer.decode(continuation, skip_special_tokens=True, clean_up_tokenization_spaces=True)

                if not self._is_valid_token(token_idx) or self._autocomplete(partial_game, overwrite_cache=True) == []:
                    scores[batch_idx][token_idx] = -torch.inf

                # elif self.verbose:
                #     print(f"Token considered valid continuation: {repr(self.tokenizer.decode(token_idx))}")

            # If we've set all the scores to -inf, then iteratively proceed through the most likely tokens until we find one that
            # is compilable
            if torch.all(scores[batch_idx] == -torch.inf):

                if self.verbose:
                    print(f"All of the top-k logits initially set to -inf, searching through all tokens")

                for idx in torch.argsort(original_scores, descending=True):
                    continuation = torch.cat((token_ids, idx.unsqueeze(0)))
                    partial_game = self.tokenizer.decode(continuation, skip_special_tokens=True, clean_up_tokenization_spaces=False)

                    if idx != self.tokenizer.eos_token_id and self._is_valid_token(idx) and self._autocomplete(partial_game, overwrite_cache=False) != []:
                        scores[batch_idx][idx] = original_scores[idx]
                        break

                # If, after all that, there's really no possible continuations, then we just set the EOS token to 0
                if torch.all(scores[batch_idx] == -torch.inf):
                    scores[batch_idx][self.tokenizer.eos_token_id] = 0
                    if self.verbose:
                        print(f"No valid tokens found, terminating sequence")

        return scores

class MixedGrammarConstraint(LogitsProcessor):
    '''
    Third attempt -- idea is to combine the autocomplete with the PICARD
    approach when we can't otherwise constrain the decoding
    '''
    def __init__(self,
                 tokenizer,
                 verbose=False):
        
        self.tokenizer = tokenizer
        self.compiler = Compile()
        self.autocomplete = Autocomplete()
        self.verbose = verbose
        self.count = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:

        self.count += 1

        # Iterate over all inputs and scores
        for batch_idx, token_ids in enumerate(input_ids):
            
            # Convert the token ids to text
            partial_game = self.tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            print(f"\n[{self.count}] Partial game: {partial_game}")

            # Obtain the next valid Ludii tokens
            # next_ludii_tokens = self.autocomplete.next_tokens(partial_game)
            try:
                next_ludii_tokens = self.autocomplete.next_tokens(partial_game)
            except SyntaxError:
                self.autocomplete = Autocomplete()
                next_ludii_tokens = self.autocomplete.next_tokens(partial_game)
                print(f"\n[SYNTAX ERROR] when processing {partial_game}")
                print(f"New autocomplete: {next_ludii_tokens}")

            # If any of the next Ludii tokens are unconstrainable primitives, then we back off
            # to the expensive compiler check on the top-k LM tokens
            if any([primitive in ludii_token for primitive in PRIMITIVES for ludii_token in next_ludii_tokens]):

                start = time.perf_counter()

                next_lm_token_indices = torch.argwhere(scores[batch_idx] > -torch.inf)

                for token_idx in next_lm_token_indices:
                    continuation = self.tokenizer.decode(torch.cat((token_ids, token_idx)), skip_special_tokens=True, clean_up_tokenization_spaces=False)

                    _, _, compilable_substring = self.compiler.compile(continuation)
                    if compilable_substring != continuation:
                        scores[batch_idx][token_idx] = -torch.inf
                    else:
                        print(f"\tToken considered valid continuation: {repr(self.tokenizer.decode(token_idx))}")

                print(f"\tNext Ludii tokens: {next_ludii_tokens}")

                print(f"[{self.count}] Time for backoff branch: {'%.3f' % (time.perf_counter() - start)}")

            # Otherwise, we form all the valid continuations as strings and re-tokenize them to determine
            # the set of valid LM tokens
            else:
                start = time.perf_counter()

                valid_token_ids = []
                for ludii_token in next_ludii_tokens:
                    continuation = partial_game + ludii_token
                    tokenized_continuation = self.tokenizer.encode(continuation)

                    # ISSUE: Consider a partial game that ends like '(merge {' -- it will have token ids ending as [313, 14634, 426]. When
                    # we consider the next valid Ludii tokens, one option is '}'. However, the game '(merge {}' will have token ids [313, 
                    # 14634, 6571] -- a mismatch, because '{}' gets merged into a single LM token despite consisting of two separate Ludii
                    # tokens. Current workaround is simply skipping tokens that would lead to such a mismatch 

                    # Ensure a tokenization match
                    if token_ids.tolist() != tokenized_continuation[:len(token_ids)]:
                        continue
                    # assert token_ids.tolist() == tokenized_continuation[:len(token_ids)],f"Tokenization mismatch -- {token_ids.tolist()} vs {tokenized_continuation[:len(token_ids)]}"
                    
                    new_token_ids = tokenized_continuation[len(token_ids):]
                    valid_token_ids.append(new_token_ids[0])

                if valid_token_ids == []:
                    raise ValueError("No valid continuations possible (almost certainly due to tokenization)")

                # Initialize all scores to -inf
                new_scores = torch.full_like(scores[batch_idx], -torch.inf)

                # Set the score of each valid token to its original score
                for token_id in valid_token_ids:
                    new_scores[token_id] = scores[batch_idx][token_id]

                scores[batch_idx] = new_scores

                print(f"[{self.count}] Time for autocomplete branch: {'%.3f' % (time.perf_counter() - start)}")

            return scores

class GrammarConstrainedLogitsProcessor(LogitsProcessor):
    def __init__(self,
                 tokenizer,
                 enforce_spacing=False,
                 verbose=False):
        
        self.tokenizer = tokenizer
        self.interface = Autocomplete()
        self.enforce_spacing = enforce_spacing
        self.verbose = verbose

        # Get the vocab and write it to a overwrite_cache file (TODO: maybe this should write a different file for each tokenizer and pass it to the java script?)
        self.vocab = tokenizer.vocab
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

        # For continuing strings, we exclude all tokens that include a quote but don't end with it
        self.continued_string_regex = re.compile(r'^[^"]*"$')
        self.valid_continuation_ids = [self.vocab[token] for token in self.vocab.keys() if not ('"' in token and not token.endswith('"'))]

    def _enforce_bracket_spacing(self, text: str):
        '''
        Enforce spacing around brackets, and remove duplicate spaces afterwards
        '''
        text = re.sub(r"([\(\)\{\}])", r" \1 ", text)
        text = re.sub(r"\s+", " ", text)
        text = text.replace(": ", ":")

        return text
    
    def _check_complete(self, text: str) -> bool:
        '''
        Returns whether all open parentheses, brackets, and braces have been closed
        '''
        
        if not text.startswith("("):
            return False

        open_paren = 0
        open_bracket = 0
        open_brace = 0

        for char in text:
            if char == "(":
                open_paren += 1
            elif char == ")":
                open_paren -= 1
            elif char == "[":
                open_bracket += 1
            elif char == "]":
                open_bracket -= 1
            elif char == "{":
                open_brace += 1
            elif char == "}":
                open_brace -= 1

        return open_paren == 0 and open_bracket == 0 and open_brace == 0


    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        '''
        Update the scores by setting the score of all 'invalid' tokens to -inf. Valid tokens are all
        tokens that can grammatically continue the current partial game, as determined by the Ludii grammar
        '''

        breakpoint()

        # Iterate over all inputs and scores
        for batch_idx, token_ids in enumerate(input_ids):
            
            # Convert the token ids to text
            sample_text = self.tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            # Temporary hack: remove the '[GAME]' prefix
            # sample_text = sample_text[6:]

            # If we've closed all the brackets, then we're done
            if self._check_complete(sample_text):
                new_scores = torch.full_like(scores[batch_idx], torch.tensor(torch.finfo(scores.dtype).min))
                new_scores[self.tokenizer.eos_token_id] = 0

                scores[batch_idx] = new_scores

                return scores

            if self.verbose:
                print("=" * 120)
                print(f"\nCurrent game: '{sample_text}'")

            # Obtain the valid next *Ludii* tokens
            try:
                next_ludii_tokens = self.interface.next_tokens(sample_text)
            except (SyntaxError, BrokenPipeError) as e:
                if self.verbose and isinstance(e, SyntaxError):
                    print(f"\n[SYNTAX ERROR] when processing {sample_text}")
                    input()

                elif self.verbose and isinstance(e, BrokenPipeError):
                    print(f"\n[BROKEN PIPE ERROR] when processing {sample_text}")
                    input()

                return scores

            if self.enforce_spacing and len(sample_text) > 1:
                next_ludii_tokens = [self._enforce_bracket_spacing(token) for token in next_ludii_tokens]

                if sample_text[-1] in ["(", ")", "{", "}"]:
                    next_ludii_tokens = [f" {token}" if not token.startswith(" ") else token for token in next_ludii_tokens]

            if self.verbose:
                print(f"\nValid Ludii tokens: {next_ludii_tokens}")

            if next_ludii_tokens == ['']:
                valid_token_ids = [self.tokenizer.eos_token_id]
            
            else:
                valid_token_ids = []
                for ludii_token in next_ludii_tokens:

                    leading_space = " " if ludii_token.startswith(" ") else ""
                    ludii_token = ludii_token.strip()

                    if ludii_token == '':
                        continue

                    if ludii_token == "NEW_STRING?":
                        valid_token_ids.append(self.tokenizer.encode(leading_space + '"')[0])

                    elif ludii_token == "NEW_INT?":
                        valid_token_ids.extend([self.tokenizer.encode(leading_space + sign + str(num))[0] for num in range(10)
                                                for sign in ["", "-"]])
                    
                    elif ludii_token == "NEW_DIM?":
                        valid_token_ids.extend([self.tokenizer.encode(leading_space + str(num))[0] for num in range(10)])

                    elif ludii_token == "NEW_FLOAT?":
                        valid_token_ids.extend([self.tokenizer.encode(leading_space + sign + str(num))[0] for num in range(10)
                                                for sign in ["", "-"]])

                    elif ludii_token == "NEW_BOOLEAN?":
                        valid_token_ids.extend([self.tokenizer.encode(leading_space + val)[0] for val in ["True", "False"]])
                        
                    # Nothing to constrain here - just gotta hope!
                    elif ludii_token == "CONTINUED_STRING?":
                        return scores
                        # valid_token_ids = self.valid_continuation_ids
                    
                    # Note: it seems that the all numbers with >= 2 digits get that many tokens
                    elif ludii_token == "CONTINUED_INT?":
                        valid_token_ids.extend([self.tokenizer.encode(leading_space + str(num))[0] for num in range(10)])

                    # It seems that dimensions are always single digits?
                    elif ludii_token == "CONTINUED_DIM?":
                        continue

                    # The possibilities here are the numerals and also '.'
                    elif ludii_token == "CONTINUED_FLOAT?":
                        valid_token_ids.extend([self.tokenizer.encode(leading_space + str(num))[0] for num in range(10)])
                        valid_token_ids.append(self.tokenizer.encode(leading_space + ".")[0])

                    elif ludii_token == "CONTINUED_BOOLEAN?":
                        # raise ValueError("Produced a CONTINUED_BOOLEAN? token -- investigate!")
                        return scores

                    # Handle cases like 'loop:NEW_BOOLEAN?' by just outputting 'loop:'
                    elif any([primitive in ludii_token for primitive in PRIMITIVES]):
                        cleaned_token = re.sub(r':.*', '', ludii_token) + ':'
                        valid_token_ids.append(self.tokenizer.encode(leading_space + cleaned_token)[0])

                    # Ostensibly the base case
                    else:
                        valid_token_ids.append(self.tokenizer.encode(leading_space + ludii_token)[0])

            if self.verbose:
                print(f"Valid token ids: {list(set(valid_token_ids))[:50]}")

            # Do some checks to see if tokenization matches up
            if self.verbose:
                relevant_ids = token_ids[3:].cpu().numpy().tolist()
                encoded = self.tokenizer.encode(sample_text, return_tensors="pt")[0].numpy().tolist()
                print(f"Tokenization match: {relevant_ids == encoded}")
                if not relevant_ids == encoded:
                    print("!!MISMATCH!!")
                    print(f"Current ids: {relevant_ids}")
                    print(f"Re-encoded ids: {encoded}")
                    input()

                top_token_ids = torch.topk(scores[batch_idx], k=len(set(valid_token_ids))).indices.cpu().numpy().tolist()
                # jaccard_similarity = len(set(valid_token_ids).intersection(set(top_token_ids))) / len(set(valid_token_ids).union(set(top_token_ids)))

                top_tokens = [self.inverse_vocab[token_id] for token_id in top_token_ids]
                valid_tokens = [self.inverse_vocab[token_id] for token_id in valid_token_ids]

                jaccard_similarity = len(set(valid_tokens).intersection(set(top_tokens))) / len(set(valid_tokens).union(set(top_tokens)))
                print(f"Jaccard similarity: {jaccard_similarity}")

            # Initialize all scores to the tensor dtype minimum
            new_scores = torch.full_like(scores[batch_idx], torch.tensor(torch.finfo(scores.dtype).min))

            # Set the score of each valid token to its original score
            for token_id in valid_token_ids:
                new_scores[token_id] = scores[batch_idx][token_id]
            
            scores[batch_idx] = new_scores
        
        return scores
