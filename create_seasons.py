import os
import pandas as pd
import pathlib
from tqdm import tqdm

path = "C:/Users/Paddy/CRT/Github/input/SEN12MS/"
filenames = list(pathlib.Path(path).glob("*/*/*.tif"))

path = "C:/Users/Paddy/CRT/Github/input/SEN12MS/ROIs1158_spring"
# s1 = fall, s6 = spring

foldernames = list(pathlib.Path(path).glob("*/"))

seasons_original = pd.read_csv('C:/Users/Paddy/CRT/Github/input/SEN12MS/seasons_spring.csv')

seasons = []
filenames = []

for folder in foldernames:
    for file in list(pathlib.Path(folder).glob("*.tif")):
        season_ref = "_".join(os.path.basename(file).split("_")[0:4])
        season = seasons_original.loc[seasons_original["scene"] == season_ref]["true_season"]
        filenames.append(os.path.basename(file))
        seasons.append(season.item())

spring = [1 if x == 'spring' else -1 for x in seasons]
summer = [1 if x == 'summer' else -1 for x in seasons]
fall = [1 if x == 'fall' else -1 for x in seasons]
winter = [1 if x == 'winter' else -1 for x in seasons]

df = pd.DataFrame(zip(filenames, spring, summer, fall, winter), columns=['scene','spring', 'summer', 'fall', 'winter'])

print("Saving data...")
df.to_csv("seasons_labeled_spring.csv", index=False)
print("Saved.")
print(df.head())