# Data

This repository contains no data. Neither licence allows it to be redistributed here:

- **WLASL** is under the Computational Use of Data Agreement (C-UDA 1.0).
- **ASL Citizen** is under the Microsoft Research License Terms: non-commercial research only.
  Nothing derived from it may be redistributed, including extracted landmarks, model checkpoints
  and video frames, and the data must be deleted when the research is complete.

This file explains how to obtain both datasets, and how to rebuild the dataset used in this project
(67 signs, 2,050 ASL Citizen clips and 791 cleaned WLASL clips). Every stage is a script in
`pipeline/`. The final vocabulary is listed [at the end](#final-vocabulary-67-signs).

**Which clips to keep:**

- **WLASL:** the 791 clips listed in [`wlasl_test_clips.csv`](wlasl_test_clips.csv) (video id,
  word, class id, source website and signer). They make up the whole WLASL test set.
- **ASL Citizen:** every clip of the 67 signs in ASL Citizen's own `splits/train.csv`, `val.csv` and
  `test.csv`, in the set that file gives it (973 / 258 / 819). Those files fully determine the
  selection, so no separate list is needed.

## Requirements

- Python 3.11, with `pip install -r model/requirements.txt remotezip yt-dlp`. The project used
  mediapipe 0.10.21, which still has the legacy Holistic API, torch 2.14, opencv-python 4.11 and
  numpy 1.26.
- `ffmpeg`, used by WLASL's preprocessing script to convert and trim the videos.
- `git` and `git-lfs`, for the Hugging Face copy of WLASL.
- About 15 GB of free disk space. The Hugging Face copy of WLASL is about 5 GB of videos, and git
  keeps a second copy of them. ASL Citizen isn't downloaded in full: only the 2,050 clips needed are
  fetched.

## 1. Obtaining the data

### WLASL (stage 01, `pipeline/01_wlasl_download/`)

1. Read and accept the C-UDA licence on the WLASL project page,
   <https://dxli94.github.io/WLASL/>. Clone the code from <https://github.com/dxli94/WLASL>.
2. Create the WLASL100 index. WLASL100 is the first 100 words of WLASL's index file
   `start_kit/WLASL_v0.3.json`: 100 words, 2,038 clips.
   ```
   cd pipeline/01_wlasl_download
   mkdir metadata
   python -c "import json; json.dump(json.load(open('<WLASL clone>/start_kit/WLASL_v0.3.json'))[:100], open('metadata/WLASL_asl100.json', 'w'))"
   ```
3. Download the clips from their original websites, then convert and trim them with WLASL's own
   scripts:
   ```
   python scripts/video_downloader.py     # -> raw_videos/
   python scripts/preprocess.py           # -> raw_videos_mp4/, then one trimmed clip per sign in videos/
   ```
4. Many of the original links no longer work. Fill in the missing clips from the Hugging Face copy
   of WLASL (this needs git-lfs):
   ```
   git clone https://huggingface.co/datasets/Voxel51/WLASL hf_raw_videos
   python scripts/extract_asl100.py       # copies only the WLASL100 clips not already in videos/
   ```
   This project ended up with 1,265 of the 2,038 WLASL100 clips. Links keep breaking, so a fresh
   download may get fewer (see [Known differences](#known-differences-from-a-fresh-run)).

### ASL Citizen (split files, needed from stage 02 on)

1. Read and accept the licence on the Microsoft Download Center page for ASL Citizen,
   <https://www.microsoft.com/en-us/download/details.aspx?id=105253> (project page:
   <https://www.microsoft.com/en-us/research/project/asl-citizen/>). The whole dataset is a single
   file, `ASL_Citizen.zip`, listed there as 42.8 GB.
2. Copy ASL Citizen's split files and licence out of the zip, without downloading the whole thing.
   Run this from the repository root:
   ```
   python -c "from remotezip import RemoteZip; z = RemoteZip('https://download.microsoft.com/download/b/8/8/b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip'); [z.extract(f, 'data/asl_citizen') for f in ('ASL_Citizen/splits/train.csv', 'ASL_Citizen/splits/val.csv', 'ASL_Citizen/splits/test.csv', 'ASL_Citizen/use.txt')]"
   ```
   This writes `data/asl_citizen/ASL_Citizen/splits/{train,val,test}.csv`, which list every clip
   with its sign, signer and ASL-LEX code. The videos come later, in stage 04, once the vocabulary
   is known.

## 2. Building the dataset with `pipeline/`

`pipeline/` has one folder per stage, numbered in the order they run. Each stage reads what earlier
stages wrote and keeps its own output in its folder. Stage 09 then assembles the final dataset in
`data/`, which is what the experiments read.

| Stage | Folder | Reads | Writes |
|---|---|---|---|
| 01 | `01_wlasl_download/` | WLASL's index and videos (section 1) | `metadata/WLASL_asl100.json` (100 words), `videos/` (1,265 WLASL100 clips) |
| — | `data/asl_citizen/` | ASL Citizen's zip (section 1) | `ASL_Citizen/splits/{train,val,test}.csv` |
| 02 | `02_vocabulary_overlap/` | 01 index, ASL Citizen splits | `class_list.txt` (the 67 classes), `vocabulary.json` |
| 03 | `03_wlasl_filter/` | 01 index and videos, 02 class list | `videos/` (843 WLASL clips of the 67 words), `wlasl_clips.json` |
| 04 | `04_asl_citizen_download/` | 02 class list, ASL Citizen splits | `videos/` (2,050 ASL Citizen clips of the 67 signs) |
| 05 | `05_split/` | 02 class list, 03 clip list, ASL Citizen splits | `split_uncleaned.json` (ASL Citizen 973 / 258 / 819, WLASL 843) |
| 06 | `06_pose_extraction/` | 03 and 04 videos | `poses_wlasl/`, `poses_asl_citizen/` (one pose file per clip) |
| 07 | `07_baseline/` | 02 class list, 05 split, 06 ASL Citizen poses | three baseline models in `experiments/checkpoints/baseline_s{0,1,2}/` |
| 08 | `08_wlasl_clean/` | 03 clip list, 05 split, 06 WLASL poses, 07 models | `first_round_baseline.json`, `first_round_per_sign.json`; the check by eye gives `excluded_clips.json` |
| 09 | `09_build_dataset/` | 02 class list, 03 and 04 videos, 03 clip list, 05 split, 08 `excluded_clips.json` | `data/split.json`, `data/class_list.txt`, `data/wlasl/`, `data/asl_citizen/videos/` |
| final poses | `06_pose_extraction/` again | `data/wlasl/videos/`, `data/asl_citizen/videos/` | `data/poses/wlasl/`, `data/poses/asl_citizen/` |

Poses are extracted twice, with the same script and settings. Stage 06 covers all 843 WLASL clips,
because stages 07 and 08 need them to decide which clips to remove. The final pass covers only the
clips in `data/`, and these are the poses the experiments use.

**Rebuilding without repeating the review:** `excluded_clips.json` is in the repository, so stages
06–08 can be skipped. Run 02 → 03 → 04 → 05 → 09, then the final pose extraction.

### Commands

Run each script from its own folder. Stage 01 and the ASL Citizen split files are in section 1.

```
cd pipeline/02_vocabulary_overlap && python find_overlap.py && cd ../..
cd pipeline/03_wlasl_filter       && python filter_wlasl.py && cd ../..
cd pipeline/04_asl_citizen_download && python download_asl_citizen.py && cd ../..
cd pipeline/05_split              && python build_split.py && cd ../..

# stages 06-08 (optional: only needed to repeat the cleaning review)
cd pipeline/06_pose_extraction
python pose_extraction.py --videos_dir ../03_wlasl_filter/videos --out_dir poses_wlasl
python pose_extraction_parallel.py --videos_dir ../04_asl_citizen_download/videos --out_dir poses_asl_citizen
cd ../07_baseline    && python train_baseline.py
cd ../08_wlasl_clean && python first_round_eval.py     # then check the flagged clips by eye
cd ../..

cd pipeline/09_build_dataset      && python build_final_dataset.py && cd ../..

# final poses, read by experiments/
cd pipeline/06_pose_extraction
python pose_extraction.py --videos_dir ../../data/wlasl/videos --out_dir ../../data/poses/wlasl
python pose_extraction_parallel.py --videos_dir ../../data/asl_citizen/videos --out_dir ../../data/poses/asl_citizen
cd ../..
```

Stage 09 ends by comparing the WLASL clips with `wlasl_test_clips.csv` and names any that differ.
Pose files hold the MediaPipe Holistic body (33 points) and both hands (21 points each), extracted
with `model_complexity=2`.

### Stage 02: choosing the vocabulary

A WLASL100 word is kept only if ASL Citizen has exactly one sign for it. ASL Citizen signs are
matched to a word by name, with or without a version number (*bed* → BED1, BED2). A word is
dropped if the matches have more than one ASL-LEX code, because the two datasets might then show
different signs for the same word.

- **Kept: 67 words**, listed [below](#final-vocabulary-67-signs).
- **Dropped, several ASL Citizen signs (31):** basketball, bed, bowling, can, candy, cheat, city,
  cool, corn, dark, deaf, decide, doctor, dog, drink, eat, fine, hat, how, jacket, kiss, letter,
  pizza, right, same, shirt, tall, walk, want, what, woman.
- **Dropped, no ASL Citizen sign found (2):** forget, short.

### Stage 08: cleaning the WLASL test clips

Accuracy is only used to decide which clips to look at. Whether a clip is removed is decided by
watching it.

1. **First-round evaluation** (`first_round_eval.py`): the three baseline models classify all 843
   WLASL clips. Accuracy is reported per source website and per sign, along with the share of frames
   in which no person is detected.
2. **By source:** clips from northtexas get 3.3% top-1, compared with at least 36.7% for every other
   source, and 31% of their frames contain no person. They were cut from compilation videos, and
   WLASL's frame ranges land on title cards, idle frames or neighbouring signs. For example, clip
   70212, labelled *book*, is mostly a title card reading "PENCIL". The whole source is removed.
3. **By sign:** the 9 signs whose accuracy is more than 1 SD below the mean (accident, before,
   fish, last, later, play, purple, school, who) are checked by eye. Each WLASL clip of these signs
   was watched and compared with ASL Citizen clips of the same sign. A clip is removed only if it
   shows a different sign.
4. **Decision:** 60 clips across the WLASL100 words, 52 of them in the 67 words, so 843 becomes 791.

| Group | Reason | WLASL video ids |
|---|---|---|
| northtexas (39) | Clips cut from compilation videos at the wrong points | 70015, 70016, 70173, 70176, 70207, 70211, 70212, 70230, 70234, 70237, 70242, 70244, 70245, 70246, 70247, 70249, 70263, 70266, 70270, 70271, 70295, 70296, 70299, 70306, 70308, 70309, 70310, 70323, 70325, 70326, 70332, 70335, 70345, 70348, 70355, 70356, 70357, 70359, 70361 |
| *before* (11) | Shows PAST (hand thrown back over the shoulder), not ASL Citizen's BEFORE | 05724, 05728, 05731, 05732, 05734, 05741, 05742, 05743, 05746, 05749, 68007 |
| *last* (4) | Shows PAST, not ASL Citizen's LAST ("final") | 32253, 32255, 32257, 32261 |
| *accident* (6) | Shows "by mistake" (hand at the chin), not ASL Citizen's ACCIDENT (collision) | 00623, 00625, 00627, 00628, 00633, 00639 |

Eight of the northtexas clips (70015, 70173, 70234, 70271, 70295, 70325, 70326, 70355) belong to
words outside the 67, so they're never in the dataset. Stage 09 reads this list from
`pipeline/08_wlasl_clean/excluded_clips.json`, which holds the same clip ids and reasons.

### Known differences from a fresh run

- **aslsignbank:** two of the 1,265 clips (65225 *book*, 65717 *fine*) came from aslsignbank. Its
  certificate doesn't match its address, so WLASL's downloader rejects it, and the clips aren't on
  Hugging Face. They were downloaded once with certificate checks turned off for that site. Without
  them a fresh run gets 1,264 clips. 65225 is a final test clip, so the WLASL test set would have
  790 clips.
- **short** is dropped as "no ASL Citizen sign found" because its ASL Citizen signs are named by
  meaning (SHORTPERSON, SHORTDISTANCE, SHORTWORD), not numbered. Either way it is dropped.

## Final vocabulary (67 signs)

Columns: the class id used by the model, the WLASL word, the matching ASL Citizen sign and its
ASL-LEX code, and the number of clips in each set. "WLASL retrieved" is the number of clips after
stage 03. "WLASL final" is the number after cleaning.

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
