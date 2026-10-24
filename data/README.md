# Data

This repository contains no data. Neither licence allows it to be redistributed here:

- **WLASL** is under the Computational Use of Data Agreement (C-UDA 1.0).
- **ASL Citizen** is under the Microsoft Research License Terms: non-commercial research only.
  Nothing derived from it may be redistributed, including extracted landmarks, model checkpoints
  and video frames, and the data must be deleted when the research is complete.

To rebuild the dataset from the original sources, follow
**[`pipeline/README.md`](../pipeline/README.md)**. The result goes in this folder.

## Which clips to keep

- **WLASL:** the 791 clips listed in [`wlasl_test_clips.csv`](wlasl_test_clips.csv) (video id,
  word, class id, source website and signer). They make up the whole WLASL test set.
- **ASL Citizen:** every clip of the 67 signs in ASL Citizen's own `splits/train.csv`, `val.csv` and
  `test.csv`, in the set that file gives it (973 / 258 / 819). Those files fully determine the
  selection, so no separate list is needed.

## Contents once built

| Path | Contents |
|---|---|
| `split.json` | Every clip → `{"subset": train / val / test (ASL Citizen) or wlasl, "action": [class_id, 1, -1]}` |
| `class_list.txt` | The 67 classes: class id, ASL Citizen sign, WLASL word |
| `wlasl/videos/` | The 791 WLASL clips, one trimmed clip per sign, `{video_id}.mp4` |
| `wlasl/wlasl_clips.json` | The same clips by word, with each clip's source website and signer |
| `asl_citizen/videos/` | The 2,050 ASL Citizen clips, named as in ASL Citizen |
| `asl_citizen/ASL_Citizen/` | ASL Citizen's split files (`splits/`) and licence (`use.txt`) |
| `poses/wlasl/`, `poses/asl_citizen/` | One MediaPipe Holistic pose file (`.pkl`) per clip |
| `wlasl_test_clips.csv` | The list of WLASL test clips (in the repository) |

## Final vocabulary (67 signs)

Columns: the class id used by the model, the WLASL word, the matching ASL Citizen sign and its
ASL-LEX code, and the number of clips in each set. "WLASL retrieved" is the number of clips after
stage 03 (step 4 of the build). "WLASL final" is the number after cleaning.

| Id | WLASL word | ASL Citizen sign | ASL-LEX | ASL Citizen train | val | test | WLASL retrieved | WLASL final |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 0 | accident | ACCIDENT | B_02_059 | 14 | 4 | 11 | 14 | 8 |
| 1 | africa | AFRICA | A_02_082 | 15 | 3 | 13 | 10 | 10 |
| 2 | all | ALL | C_03_088 | 15 | 3 | 13 | 14 | 14 |
| 3 | apple | APPLE | A_03_054 | 17 | 4 | 11 | 13 | 12 |
| 4 | before | BEFORE | G_02_053 | 14 | 5 | 11 | 22 | 10 |
| 5 | bird | BIRD | A_01_041 | 14 | 4 | 13 | 11 | 11 |
| 6 | birthday | BIRTHDAY | D_01_054 | 15 | 4 | 12 | 9 | 7 |
| 7 | black | BLACK | B_02_006 | 15 | 4 | 12 | 16 | 15 |
| 8 | blue | BLUE | C_02_062 | 15 | 3 | 13 | 13 | 13 |
| 9 | book | BOOK | A_01_027 | 15 | 4 | 12 | 13 | 11 |
| 10 | brown | BROWN | D_03_045 | 13 | 5 | 12 | 10 | 9 |
| 11 | but | BUT | D_03_030 | 15 | 3 | 12 | 8 | 8 |
| 12 | chair | CHAIR | A_01_017 | 14 | 4 | 12 | 12 | 10 |
| 13 | change | CHANGE | E_01_031 | 13 | 5 | 11 | 13 | 13 |
| 14 | clothes | CLOTHES | E_03_096 | 15 | 4 | 12 | 10 | 10 |
| 15 | color | COLOR | C_02_039 | 13 | 4 | 13 | 9 | 9 |
| 16 | computer | COMPUTER | A_01_054 | 15 | 4 | 11 | 21 | 21 |
| 17 | cook | COOK | D_03_078 | 14 | 4 | 13 | 9 | 9 |
| 18 | cousin | COUSIN | C_01_023 | 14 | 4 | 13 | 16 | 15 |
| 19 | cow | COW | A_01_033 | 13 | 5 | 12 | 10 | 10 |
| 20 | dance | DANCE | B_02_063 | 15 | 3 | 14 | 8 | 8 |
| 21 | enjoy | ENJOY | D_03_033 | 15 | 3 | 11 | 8 | 8 |
| 22 | family | FAMILY | C_01_025 | 15 | 3 | 12 | 15 | 15 |
| 23 | finish | FINISH | B_03_025 | 15 | 3 | 14 | 16 | 15 |
| 24 | fish | FISH | A_03_040 | 14 | 4 | 13 | 15 | 15 |
| 25 | full | FULL | D_02_048 | 15 | 3 | 11 | 10 | 10 |
| 26 | give | GIVE | F_01_023 | 15 | 4 | 11 | 10 | 10 |
| 27 | go | GO | C_03_056 | 15 | 3 | 13 | 18 | 18 |
| 28 | graduate | GRADUATE | B_03_081 | 15 | 4 | 12 | 14 | 13 |
| 29 | hearing | HEARING | C_02_023 | 14 | 4 | 14 | 14 | 13 |
| 30 | help | HELP | D_01_042 | 13 | 5 | 13 | 15 | 15 |
| 31 | hot | HOT | F_02_093 | 16 | 4 | 11 | 17 | 16 |
| 32 | language | LANGUAGE | A_01_067 | 15 | 4 | 11 | 10 | 10 |
| 33 | last | LAST | B_01_038 | 14 | 4 | 13 | 12 | 8 |
| 34 | later | LATER | D_01_077 | 16 | 4 | 12 | 12 | 12 |
| 35 | like | LIKE | F_03_063 | 15 | 3 | 12 | 18 | 17 |
| 36 | man | MAN | C_01_040 | 15 | 4 | 12 | 13 | 13 |
| 37 | many | MANY | D_03_065 | 13 | 6 | 12 | 15 | 14 |
| 38 | medicine | MEDICINE | C_03_040 | 15 | 3 | 13 | 8 | 8 |
| 39 | meet | MEET | D_02_018 | 16 | 3 | 10 | 10 | 10 |
| 40 | mother | MOTHER | B_02_008 | 14 | 3 | 15 | 18 | 18 |
| 41 | need | NEED | C_02_034 | 15 | 3 | 12 | 8 | 7 |
| 42 | no | NO | C_03_041 | 16 | 3 | 11 | 17 | 17 |
| 43 | now | NOW | C_03_062 | 15 | 3 | 12 | 16 | 15 |
| 44 | orange | ORANGE | A_02_020 | 14 | 4 | 13 | 17 | 15 |
| 45 | paint | PAINT | D_01_048 | 15 | 3 | 11 | 7 | 7 |
| 46 | paper | PAPER | A_01_026 | 14 | 4 | 13 | 9 | 8 |
| 47 | pink | PINK | C_01_074 | 14 | 4 | 13 | 10 | 9 |
| 48 | play | PLAY | C_01_066 | 14 | 4 | 14 | 11 | 11 |
| 49 | pull | PULL | D_02_062 | 12 | 5 | 13 | 9 | 9 |
| 50 | purple | PURPLE | B_02_067 | 14 | 4 | 13 | 9 | 8 |
| 51 | school | SCHOOL | C_03_089 | 16 | 3 | 11 | 10 | 9 |
| 52 | secretary | SECRETARY | A_03_059 | 15 | 4 | 12 | 10 | 10 |
| 53 | son | SON | B_02_086 | 15 | 4 | 12 | 11 | 10 |
| 54 | study | STUDY | G_01_090 | 15 | 3 | 12 | 10 | 10 |
| 55 | table | TABLE | A_03_025 | 15 | 4 | 11 | 9 | 8 |
| 56 | tell | TELL | C_02_013 | 15 | 3 | 12 | 11 | 11 |
| 57 | thanksgiving | THANKSGIVING | D_01_047 | 15 | 4 | 12 | 16 | 16 |
| 58 | thin | THIN | C_02_002 | 13 | 4 | 13 | 18 | 18 |
| 59 | thursday | THURSDAY | B_02_045 | 13 | 5 | 12 | 12 | 12 |
| 60 | time | TIME | B_02_080 | 15 | 4 | 12 | 10 | 9 |
| 61 | white | WHITE | E_03_046 | 14 | 5 | 11 | 12 | 11 |
| 62 | who | WHO | C_01_041 | 14 | 4 | 14 | 18 | 18 |
| 63 | work | WORK | B_03_059 | 14 | 3 | 15 | 11 | 11 |
| 64 | wrong | WRONG | B_01_055 | 13 | 5 | 12 | 10 | 9 |
| 65 | year | YEAR | B_02_015 | 15 | 4 | 12 | 15 | 14 |
| 66 | yes | YES | G_03_074 | 15 | 5 | 10 | 18 | 18 |
| | **Total** | | | **973** | **258** | **819** | **843** | **791** |

The retrieved and final counts differ for some signs that weren't in the by-sign check (for
example *birthday*, 9 → 7). Those clips are northtexas clips, which were removed as a whole source.
