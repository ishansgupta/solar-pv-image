import pandas as pd
import sys

# Merges the two prediction csv files by averaging the two scores.
# Usage: python3 merge_submissions.py <file 1> <file 2> <output file>

one = pd.read_csv(sys.argv[1])
two = pd.read_csv(sys.argv[2])

one.score = (one.score + two.score) / 2

one.to_csv(sys.argv[3])
