from dataclasses import dataclass
from enum import Enum
import typing

@dataclass
class ArchiveGame:
    game_str: str
    fitness_score: float
    evaluation: typing.Dict[str, float]
    lineage: list
    generation: int
    original_game_name: str
    epoch: int

class EliteSelectionStrategy(str, Enum):
    RANDOM = "random"
    UCB = "ucb"

class MutationSelectionStrategy(str, Enum):
    RANDOM = "random"
    UCB_DEPTH = "ucb_depth"
    UCB_LUDEME = "ucb_ludeme"

class MutationStrategy(str, Enum):
    STANDARD = "standard"
    GRAMMAR_ENFORCED = "grammar_enforced"

class FitnessEvaluationStrategy(str, Enum):
    RANDOM = "random"
    UCT = "uct"
    ONE_PLY = "one_ply"
    COMBINED = "combined"

class FitnessAggregationFn(str, Enum):
    MEAN = "mean"
    HARMONIC_MEAN = "harmonic_mean"
    MIN = "min"


VALIDATION_GAMES = [
    # "Alquerque", # too long to evaluate (>100 moves per player)
    "Amazons",
    "ArdRi",
    "Ataxx",
    # "Bao Ki Arabu (Zanzibar 1)", # Mancala game
    "Breakthrough",
    # "Chess", # too long description
    # "English Draughts", # too long to evaluate (>100 moves per player)
    # "Fanorona", # too long description
    "Gomoku",
    "Havannah",
    "Hex",
    "Knightthrough",
    "Konane",
    "Pretwa",
    "Reversi",
    "Shobu",
    "Tablut",
    "Tron", # added by Graham
    # "XII Scripta", # too long to evaluate (>100 moves per player)
    "Yavalath"
]