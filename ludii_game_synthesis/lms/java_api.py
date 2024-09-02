from __future__ import annotations

import os
import pathlib
import re
import select
import subprocess
import time

from lms.java_helpers import *

default_bytecode_path = os.path.join(pathlib.Path(__file__).parent.resolve(), 'ludii_fork/')


class JavaInterface():
    """
    A class for interfacing with the Java code. This class starts the Java process and communicates with it via stdin/stdout.
    """

    def __init__(self,
                 classpath: str,
                 name: str,
                 bitecode_path: str = default_bytecode_path,
                 dependencies: list = BASE_DEPENDENCIES,
                 verbose=False):

        self.name = name
        self.verbose = verbose
        self.restart_time = 0
        self.java_process: subprocess.Popen | None = None

        self.command = [
            "java",
            "-cp",
            ':'.join(map(lambda d: bitecode_path + d, dependencies)),
            classpath,
        ]

        if self.verbose:
            print("Java command: " + ' '.join(self.command))

        self._restart()

    def query(self, inp: str, timeout=-1) -> str:

        try:
            self._write(inp)
            response = self._read(timeout=timeout)
        except BrokenPipeError as e:
            print(f"The following input triggered a broken pipe for {self.name}: \n{inp}")
            raise e


        if response == '':
            if self.verbose:
                print(f"Java process terminated unexpectedly with return code {self.java_process.returncode}. Restarting...")
            self._restart()
            return self.query(inp)

        return response

    def _restart(self):
        self._terminate()

        if time.time() - self.restart_time > 0.5:
            self.java_process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.restart_time = time.time()

            if self.verbose:
                print(f'Started java {self.name} process...')

            while True:
                line = self._read()
                if self.verbose: print(line)
                if "Ready" in line:
                    break
        else:
            raise Exception("Java process restarted too quickly. Something is very wrong.")

    def _read(self, timeout=-1):
        """
        Read data from stdout with a timeout.
        :param timeout: The maximum time to wait for a response in seconds.
        :return: The response from Java if available, otherwise raise TimeoutError.
        """
        ready = True
        if timeout > 0:
            ready, _, _ = select.select([self.java_process.stdout], [], [], timeout)

        if ready:
            return self.java_process.stdout.readline().decode("utf-8").replace("\n", "").replace("\\n", "\n")
        else:
            self._terminate()
            raise TimeoutError(f"Timeout exceeded ({timeout}s) while waiting for response from Java process.")

    def _write(self, message):
        message = message.replace("\n", "\\n")

        self.java_process.stdin.write(f"{message}\n".encode("utf-8"))
        self.java_process.stdin.flush()



    def _terminate(self):
        if self.java_process is not None:
            if self.verbose:
                print("Terminating java process...")
            self.java_process.stdin.close()
            self.java_process.terminate()
            # self.java_process.wait(timeout=0.2)

    def __del__(self):
        self._terminate()


class Autocomplete(JavaInterface):
    """
    This endpoint takes a partial game string and returns a list of the next possible ludemes.
    Each item contains the remaining characters to finish typing a valid ludeme.
    """

    def __init__(self, bitecode_path: str = default_bytecode_path, verbose=False):
        super().__init__('approaches.symbolic.api.AutocompleteEndpoint', 'autocomplete', bitecode_path=bitecode_path, verbose=verbose)

    def next_tokens(self, game_string, overwrite_cache=True):
        if not overwrite_cache:
            game_string = '@' + game_string
        completions = self.query(game_string).split("||")
        return list(map(lambda c: c.split("|")[0], completions))


class Compile(JavaInterface):
    """
    This endpoint takes a partial game string and returns:
    1. 1 if the game is valid, 0 if it isn't
    2. A fast evaluation of the game between 0 and 1. If the game is invalid, this is 0.
    3. The syntactically correct substring. If the game is valid, this is the entire game string.
    4. The missing requirements for the game to compile.
    5. The crash messages if individual ludemes will lead to a crash
    """

    def __init__(self, bitecode_path: str = default_bytecode_path, verbose=False):
        super().__init__('approaches.symbolic.api.FractionalCompilerEndpoint', 'compiler', bitecode_path=bitecode_path, verbose=verbose)

    def compile(self, game_string, overwrite_cache=True):
        if not overwrite_cache:
            game_string = '@' + game_string

        results = self.query(game_string).split("||")
        return {"compiles": results[0] == "1", "evaluation": float(results[1]), "valid_substring": results[2],
                "missing_requirements": results[3], "crash_messages": results[4]}


class SkillTrace(JavaInterface):
    """
    This endpoint takes a complete game string and returns the skill trace evaluation.
    NOTE: This endpoint can be slow. Expect 5 minutes for a game like Hex.
    """

    def __init__(self, bitecode_path: str = default_bytecode_path, verbose=False):
        super().__init__('approaches.symbolic.api.evaluation.SkillTraceEndpoint', 'skill trace', bitecode_path=bitecode_path, verbose=verbose)

    def evaluate(self, game_string):
        return self.query(game_string)


class StandardEvaluation(JavaInterface):
    """
    This endpoint takes a complete game string and returns its balance, completion rate, drawishness, and mean number of turns.
    Returns -1 if the game does not compile. Returns -2 if the game can't be played.
    """

    def __init__(self, bitecode_path: str = default_bytecode_path, verbose=False):
        super().__init__('approaches.symbolic.api.evaluation.StandardEvaluationEndpoint', 'random evaluation',
                         bitecode_path=bitecode_path, verbose=verbose)

    def evaluate(self,
                 game_string: str, 
                 ai_name: str = "Random",
                 num_games: int = 20,
                 thinking_time: float = 0.1, 
                 max_turns: int = 500,
                 timeout: int = -1
                 ):
        '''
        Evaluate the provided game using one of the Ludii AIs. Arguments determine:
        - the AI used
        - the number of games played for the evaluation
        - the thinking time given to the AI each turn
        - the maximum number of turns allowed in a game before it is called a draw
        '''
        
        evaluation = {"compilable": True, "playable": True, "balance": -1, "completion": -1, "drawishness": -1,
                      "mean_turns": -1, "decision_moves": -1, "board_coverage_default": -1, "wins": []}

        raw_evaluation = self.query(f"{ai_name}|{num_games}|{thinking_time}|{max_turns}|{game_string}", timeout=timeout).split("||")

        if raw_evaluation == ["-1"]:
            evaluation["compilable"] = False
            evaluation["playable"] = False

        elif raw_evaluation == ["-2"]:
            evaluation["playable"] = False

        elif len(raw_evaluation) == 1:
            evaluation["compilable"] = False
            evaluation["playable"] = False
            print(f"Encountered unexpected output from StandardEvaluation endpoint: {raw_evaluation}")

        else:
            metrics, wins = raw_evaluation[0].split("|"), raw_evaluation[1].split("|")
            balance, completion, drawishness, mean_turns, decision_moves, board_coverage_default = metrics

            evaluation["balance"] = float(balance)
            evaluation["completion"] = float(completion)
            evaluation["drawishness"] = float(drawishness)
            evaluation["mean_turns"] = float(mean_turns)
            evaluation["decision_moves"] = float(decision_moves)
            evaluation["board_coverage_default"] = float(board_coverage_default)

            evaluation["wins"] = list(map(int, wins))

        return evaluation

    def two_step_evaluate(self,
                          game_string: str,
                          ai_name: str = "Random",
                          num_games: int = None,
                          thinking_time: float = None,
                          max_turns: int = None,
                          min_random_balance: float = 0.5,
                          min_random_decisions: float = 0.5):

        # Perform a quick random evaluation to check for certain minimal criteria
        random_evaluation = self.evaluate(game_string, "Random", 100, 1, 500)

        if not random_evaluation["compilable"] or not random_evaluation["playable"]:
            return random_evaluation

        # If the game fails to meet the minimal criteria under a random evaluation, we enforce a return that will
        # assign the game the UNINTERESTING_FITNESS score
        if random_evaluation["balance"] < min_random_balance or random_evaluation["decision_moves"] < min_random_decisions:
            evaluation = {"compilable": True, "playable": True, "trace_score": 0, "balance": 0, "completion": 0, "drawishness": 0,
                          "mean_turns": 0, "decision_moves": 0, "board_coverage_default": 0, "wins": [], "error": "Uninteresting under random eval"}
            
            return evaluation


        true_evaluation = self.evaluate(game_string, ai_name, num_games, thinking_time, max_turns)

        return true_evaluation


class FastTrace(JavaInterface):
    """
    This endpoint takes a complete game string and evaluates how often a strong AI can win against random play.
    This evaluation occurs multiple times setting the strong AI to be different players. For example, by default, for a
     game of Hex it will play 20 times. 10 times the strong AI will be player 1 and 10 times it will be player 2.
    """

    def __init__(self, bitecode_path: str = default_bytecode_path, verbose=False):
        super().__init__('approaches.symbolic.api.evaluation.FastTraceEndpoint', 'fast trace evaluation', bitecode_path=bitecode_path, verbose=verbose)

    def evaluate(self, game_string, thinking_time=0.0025, max_time=20, max_turns=500, trials_per_player=10):
        result = self.query(game_string + f"|{thinking_time}|{max_time}|{max_turns}|{trials_per_player}")
        return float(result)


class Novelty(JavaInterface):
    """
    This endpoint takes a partial game string and returns:
    1. 1 if the game is valid, 0 if it isn't
    2. A fast evaluation of the game between 0 and 1. If the game is invalid, this is 0.
    3. The compilable substring. If the game is valid, this is the entire game string.
    """

    def __init__(self, bitecode_path: str = default_bytecode_path, verbose=False):
        super().__init__('approaches.symbolic.api.evaluation.NoveltyEndpoint', 'novelty', bitecode_path=bitecode_path, verbose=verbose)
        self.initialized = False

    def load_all_games(self):
        result = self.query("|")
        assert result == "0", "Failed to load all games"
        self.initialized = True

    def load_game_library(self, game_descriptions: list):
        assert len(game_descriptions) > 1, "For some reason we need at least 2 games to compare against"
        result = self.query("|".join(game_descriptions))
        assert result == "0", "Failed to load game library"
        self.initialized = True

    def evaluate(self, game_string):
        if not self.initialized:
            raise Exception("Novelty endpoint has not been initialized with a game library")
        return float(self.query(game_string))


class ClassPaths(JavaInterface):
    """
    This endpoint takes a partial game string and returns:
    1. 1 if the game is valid, 0 if it isn't
    2. A fast evaluation of the game between 0 and 1. If the game is invalid, this is 0.
    3. The compilable substring. If the game is valid, this is the entire game string.
    """

    def __init__(self, bitecode_path: str = default_bytecode_path, verbose=False):
        super().__init__('approaches.symbolic.api.ClassPathEndpoint', 'classpath', bitecode_path=bitecode_path, verbose=verbose)

    def compile(self, game_string):
        return self.query(game_string).split(" ")


class Concepts(JavaInterface):
    """
    This endpoint takes a complete (TODO partial) game string and returns:
    1. The boolean concepts
    2. The non-boolean concepts
    """

    def __init__(self, bitecode_path: str = default_bytecode_path, verbose=False):
        super().__init__('approaches.symbolic.api.CompleteConceptEndpoint', 'concept', bitecode_path=bitecode_path, verbose=verbose)

    def compile(self, game_string):
        query_str = self.query(game_string)
        if query_str == "-1":
            return []

        bool_str, non_bool_str = query_str.split("|")

        all_concepts = [bit == '1' for bit in bool_str]

        for idx in range(len(all_concepts)):
            if CONCEPT_DTYPES[idx] != bool:
                all_concepts[idx] = None

        for string in non_bool_str[1:-1].split(", "):
            idx, value = string.split("=")
            idx = int(idx)
            assert not all_concepts[idx]

            try:
                all_concepts[idx] = CONCEPT_DTYPES[idx](value)
            except ValueError:
                all_concepts[idx] = CONCEPT_DTYPES[idx](float(value))

        return all_concepts


class HeadlessInterface(JavaInterface):
    """
    WIP: headless mode for interfacing with Ludii GUI
    """

    def __init__(self, game, width, height, bitecode_path: str = default_bytecode_path, verbose=False):
        super().__init__(classpath='app.headless.HeadlessEndpoint',
                         name='headless',
                         bitecode_path=bitecode_path,
                         dependencies=HEADLESS_DEPENDENCIES,
                         verbose=verbose)

        self.query(f"setup|{width}|{height}|{game}")

    def click(self, x, y):
        return self.query(f"click|{x}|{y}")


class WebAppInterface(JavaInterface):
    """
    WIP: Web App mode for interfacing with Ludii GUI
    """

    def __init__(self, game, width, height, bitecode_path: str = default_bytecode_path, verbose=False):
        super().__init__(classpath='app.WebAppEndpoint',
                         name='webapp',
                         bitecode_path=bitecode_path,
                         dependencies=HEADLESS_DEPENDENCIES,
                         verbose=verbose)

        self.query(f"setup|{width}|{height}|{game}")

    def click(self, x, y):
        path, player, game_over, winner = self.query(f"click|{x}|{y}").split('|')

        return {"path": path, "player": player, "game_over": game_over, "winner": winner}



if __name__ == '__main__':

    hex_game = open("./ludii_data/games/expanded/board/space/connection/Hex.lud", "r").read()
    complete_game = open("./ludii_data/games/expanded/board/space/line/Tic-Tac-Toe.lud", "r").read()
    test_game = """(game "GAME_NAME" 
    (players 2)

    (equipment {
            (mancalaBoard 4 9 store:None {
                    (track "Track1" "0,E,N1,W" loop:True P1)
                    (track "Track2" "18,E,N1,W" loop:True P2)
            })
            (regions "Home" P1 
                (sites Track "Track1")
            )
            (regions "Home" P2 
                (sites Track "Track2")
            )
            (regions "Inner" P1 
                (difference 
                    (sites Track "Track1")
                    (sites Bottom)
                )
            )
            (piece "PIECE_ALPHA" Shared)
    })

    (rules 
        (start 
            (set Count 2 to:
                (union 
                    (sites P1 "Home")
                    (sites P2 "Home")
                )
            )
        )
        (play 
            (or 
                (move Pass)
                (move Select 
                    (from (sites Mover "Home") if:(< 0 
                            (count at:(from))
                    ))
                    (then (sow "Track" owner:
                            (mover)
                            apply:
                            (if (and {
                                        (!= 
                                            (count at:(if (is Mover P1) (+ (to) (* 9 2)) (- (to) (* 9 2))))
                                        0)
                                        (!= 
                                            (count at:(if (is Mover P1) (+ 9  9) (- (to) 9)))
                                        0)
                                        (is In (to) (sites Mover "Inner"))
                                        }) (and {
                                        (fromTo (from (if (is Mover P1) (+ (to) (* 9 2)) (- (to) (* 9 2)))) (to (to)) count:(count at:(if (is Mover P1) (+ (to) (* 9 2))  (- (to) (* 9 2)) )))
                                        (fromTo  (from (if (is Mover P1) (+ (to) 9)  (- (to) 9) ))   (to (to))  count:1 )
                                        (sow 
                                            (to)
                                            count:
                                            (+ (count at:(if (is Mover P1) (+ (to) 9) (- (to) 9))) (count at:(if (is Mover P1) (+ (to) (* 9 2)) (- (to) (* 9 2)))))
                                            "Track" owner:
                                            (mover)
                                        )
                            }))
                    ))
                )
            )
        )

        (end {
                (if (= (count in:(sites Next "Home")) 0) (result Mover Win))
                (if (was Pass) (result Next Win))
        })
    )
)""".replace("\n", " ")
    
    # from tqdm import tqdm
    # import numpy as np
    #
    # trace = FastTrace()
    # evaluations = []
    # for _ in tqdm(range(1), desc="fast trace"):
    #     evaluation = trace.evaluate(hex_game, 0.01, 60)
    #     evaluations.append(evaluation)
    #
    # print(f"Mean: {np.mean(evaluations)}")
    # print(f"Std: {np.std(evaluations)}")
    # breakpoint()

    # Test classpath endpoint
    # print(Autocomplete().next_tokens(complete_game))
    # print("ClassPaths:", ClassPaths(verbose=True).compile(complete_game))

    # Test Concepts endpoint
    # concepts = Concepts()
    # print(concepts.compile(complete_game))
    # print(concepts.compile(hex_game))
    # for i in range(5):
    #     print(len(concepts.compile(test_game)))
    # print(CONCEPT_NAMES[4])  # Easy way to get the name of the 4th concept

    # Test Fractional Compiler endpoint
    # partial_dice_chess = '(game "Dice Chess" (players {(player N) (player S)}) (equipment {(board (square 8)) (piece "Pawn" Each (or {(if (is In (from) (sites Start (piece (what at:(from))))) (move Slide Forward (between (exact 2) if:(is Empty (between))) (to if:(is Empty (to))))) (move Step Forward (to if:(is Empty (to)))) (move Step (directions {FR FL}) (to if:(is Enemy (who at:(to))) (apply (remove (to)))))} (then (if (is In (last To) (sites Mover "Promotion")) (moveAgain))))) (piece "Bishop" Each (move Slide Diagonal (to if:(is Enemy (who at:(to))) (apply (remove (to)))))) (piece "Knight" Each (move Leap {{F F R F} {F F L F}} (to if:(not (is Friend (who at:(to)))) (apply (if (is Enemy (who at:(to))) (remove (to))))))) (piece "Rook" Each (move Slide Orthogonal (to if:(is Enemy (who at:(to))) (apply (remove (to)))))) (piece "Queen" Each (move Slide (to if:(is Enemy (who at:(to))) (apply (remove (to)))))) (piece "King_noCross" Each (move Step (to if:(not (is Friend (who at:(to)))) (apply (if (is Enemy (who at:(to))) (remove (to))))))) (regions "Promotion" P1 (sites Top)) (regions "Promotion" P2 (sites Bottom))}) (rules (start {(place "Pawn1" {"A2" "B2" "C2" "D2" "E2" "F2" "G2" "H2"}) (place "Pawn2" {"H7" "G7" "E7" "F7" "D7" "C7" "B7" "A7"}) (place "Bishop1" {"C1" "F1"}) (place "Bishop2" {"C8" "F8"}) (place "Knight1" {"B1" "G1"}) (place "Knight2" {"G8" "B8"}) (place "Rook1" {"A1" "H1"}) (place "Rook2" {"H8" "A8"}) (place "Queen1" coord:"D1") (place "Queen2" coord:"D8") (place "King_noCross1" coord:"E1") (place "King_noCross2" coord:"E8")}) (play (do (if (not (is Prev Mover)) (roll)'
    # fractional_compiler = Compile()
    # print(fractional_compiler.compile(partial_dice_chess)["missing_requirements"])
    # print(fractional_compiler.compile(partial_dice_chess.replace("(roll)", ""))["missing_requirements"])

    # Tests Random Evaluation endpoint
    # print(StandardEvaluation().evaluate(test_game, num_games=1000))
    # print(StandardEvaluation().evaluate(test_game, ai_name="UCT", num_games=3, thinking_time=0.025, max_turns=100))

    # print("Starting random evals stress test...")

    # for i in range(10):
    #     random_evaluator = RandomEvaluation()
    #     for j in range(10):
    #         if j % 5 == 0:
    #             print(random_evaluator.scalar_evaluate("Crash"))
    #         else:
    #             print(random_evaluator.scalar_evaluate(complete_game))
    #
    #     # random_eval.terminate()
    #     print(i, psutil.virtual_memory().percent, psutil.Process().memory_info().rss / (1024 * 1024))
    #
    # print("Done!")

    # Test Fast Trace endpoint
    fastTrace = FastTrace()
    print("Fast Trace:", fastTrace.evaluate(complete_game, 0.01, 60))

    # Novelty
    # novelty = Novelty(verbose=True)
    # print("starting")
    # # novelty.load_all_games()
    # novelty.load_game_library([complete_game, hex_game])
    # print("Novelty:", novelty.evaluate(complete_game))
    # print("Novelty:", novelty.evaluate(hex_game))
    # print("Novelty:", novelty.evaluate(test_game))

    # time
    # randomEval = RandomEvaluation()
    # autocomplete = Autocomplete()
    # for i in range(10 ** 5):
    #     start = time.time_ns()
    #     # randomEval.evaluate(complete_game)
    #     autocomplete.next_tokens('(game "Hello" ')
    #     end = time.time_ns()
    #
    #     if i % 500 == 0:
    #         print(i, (end - start) / 10 ** 9)
