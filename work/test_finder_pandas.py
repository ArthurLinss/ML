import pandas as pd
import re

class AutoFind:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def find_something(self, target_strings: list, something, distance=3) -> pd.DataFrame:
        results = []
        info = ""
        target_strings = list(set(target_strings))

        for index, row in self.dataframe.iterrows():
            for col_index, cell in enumerate(row):
                for target_string in target_strings:
                    if cell == target_string:
                        row_index = index
                        for i in range(max(0, row_index - distance), min(len(self.dataframe), row_index + distance + 1)):
                            for j in range(max(0, col_index - distance), min(len(row), col_index + distance + 1)):
                                if isinstance(something, str):
                                    if self.dataframe.iloc[i, j] == something:
                                        dist_y = (i - row_index)
                                        dist_x = (j - col_index)
                                        dist = abs(dist_x) + abs(dist_y)
                                        col_dist = abs(j - col_index)
                                        # Adjusting probability based on distance and position relative to target_string
                                        if i == row_index and j >= col_index:  # Lower probability for cells below target_string
                                            probability = 1
                                            info = "right same row"
                                        elif i == row_index and j < col_index:  # Lower probability for cells below target_string
                                            probability = 0.1
                                            info = "left same row"
                                        elif i >= row_index and j == col_index:  # Lower probability for cells below target_string
                                            probability = 0.95
                                            info = "below same col"
                                        elif i < row_index and j == col_index:  # Lower probability for cells below target_string
                                            probability = 0.11
                                            info = "above same col"
                                        else:
                                            probability = 0.1
                                            info = "other"

                                        probability = probability / (dist)

                                        results.append({
                                            'row_index %s' % target_string: row_index,
                                            'col_index %s' % target_string: col_index,
                                            'row %s' % something:i,
                                            "col %s" % something:j,
                                            "dist" : dist,
                                            "distx" : dist_x,
                                            "disty" : dist_y,
                                            'probability': probability,
                                            "info":info
                                        })
                                elif isinstance(something, re.Pattern):
                                    if something.match(str(self.dataframe.iloc[i, j])):
                                        dist_y = (i - row_index)
                                        dist_x = (j - col_index)
                                        dist = abs(dist_x) + abs(dist_y)
                                        col_dist = abs(j - col_index)
                                        # Adjusting probability based on distance and position relative to target_string
                                        if i == row_index and j >= col_index:  # Lower probability for cells below target_string
                                            probability = 1
                                            info = "right same row"
                                        elif i == row_index and j < col_index:  # Lower probability for cells below target_string
                                            probability = 0.1
                                            info = "left same row"
                                        elif i >= row_index and j == col_index:  # Lower probability for cells below target_string
                                            probability = 0.95
                                            info = "below same col"
                                        elif i < row_index and j == col_index:  # Lower probability for cells below target_string
                                            probability = 0.11
                                            info = "above same col"
                                        else:
                                            probability = 0.1
                                            info = "other"

                                        if dist>0:
                                            probability = probability / (dist)
                                        else:
                                            probability = 0

                                        results.append({
                                            'row_index %s' % target_string: row_index,
                                            'col_index %s' % target_string: col_index,
                                            'row %s' % something:i,
                                            "col %s" % something:j,
                                            "dist" : dist,
                                            "distx" : dist_x,
                                            "disty" : dist_y,
                                            'probability': probability,
                                            "info":info
                                        })

        return pd.DataFrame(results)

# Example usage:
# Assuming df is your DataFrame
data = {
    '0': ['apple', 'banana', 'cat', 'dog', 'elephant'],
    '1': ['dog', 'elephant', 'frog', 'apple', 'banana'],
    '2': ['cat', 'dog', 'apple', 'banana', 'elephant'],
    '3': ['frog', 'cat', 'dog', 'apple', 'dog'],
    '4': ['apple', 'banana', 'cat', 'banana', 'elephant']
}

df = pd.DataFrame(data)

# Creating an instance of AutoFind
finder = AutoFind(df)

# Finding something with string input
dfres_str = finder.find_something(['apple'], 'dog', distance=2)

# Finding something with regex input
pattern = re.compile(r'\d{6}')  # Regex pattern for a 6-digit number
pattern = re.compile(r'^.{3}$') # string lentght 3

dfres_regex = finder.find_something(['apple'], pattern, distance=2)

if "probability" in dfres_str:
    dfres_str = dfres_str[dfres_str["probability"] > 0.8]

if "probability" in dfres_regex:
    dfres_regex = dfres_regex[dfres_regex["probability"] > 0.8]

print("String search results:")
print(dfres_str)

print("\nRegex search results:")
print(dfres_regex)
