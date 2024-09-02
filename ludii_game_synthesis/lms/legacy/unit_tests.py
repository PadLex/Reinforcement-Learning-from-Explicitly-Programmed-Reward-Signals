import os
import unittest

from java_api import *

KEY_GAME_NAMES = ["Hex", "Tic-Tac-Toe", "Chess", "Go"]
ALL_GAMES = []
KEY_GAMES = []
# recursive search for all files in the games directory
for root, dirs, files in os.walk("./ludii_data/games/expanded"):
    for file in files:
        if file.endswith(".lud"):
            path = os.path.join(root, file)
            with open(path, "r") as f:
                game = f.read()
                ALL_GAMES.append(game)
                if file[:-4] in KEY_GAME_NAMES:
                    KEY_GAMES.append(game)

assert len(KEY_GAMES) == len(KEY_GAME_NAMES), f"Expected {len(KEY_GAME_NAMES)} key games, found {len(KEY_GAMES)}"

EXPECTED_RESULTS = {}
if os.path.exists("expected_results.json"):
    with open("expected_results.json", "r") as f:
        EXPECTED_RESULTS = json.load(f)

class JavaAPI(unittest.TestCase):
    def test_autocomplete_unchanged(self):
        autocomplete = Autocomplete()

        if "autocomplete" in EXPECTED_RESULTS:
            for game in KEY_GAMES:
                overwrite = False
                if game not in EXPECTED_RESULTS["autocomplete"]:
                    EXPECTED_RESULTS["autocomplete"][game] = []
                    overwrite = True

                for i in range(len(game)):
                    if overwrite:
                        EXPECTED_RESULTS["autocomplete"][game].append(autocomplete.next_tokens(game[:i]))
                    else:
                        self.assertEqual(autocomplete.next_tokens(game[:i]), EXPECTED_RESULTS["autocomplete"][game][i])


if __name__ == '__main__':
    unittest.main()

    # Overwrite the expected results
    with open("expected_results.json", "w") as f:
        print("Overwriting expected results...", EXPECTED_RESULTS)
        json.dump(EXPECTED_RESULTS, f, indent=4)
