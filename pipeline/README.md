# Building the dataset

These steps rebuild the dataset used in this project from the original sources: 67 signs, 2,050
ASL Citizen clips and 791 cleaned WLASL clips. Each step runs one stage of `pipeline/`, from its own
folder (`01_wlasl_download/`, `02_vocabulary_overlap/`, …), and each stage reads what the earlier
ones wrote. The result goes in `data/`, described in [`data/README.md`](../data/README.md) along with
the list of clips to keep and the final vocabulary.

The vocabulary comparison (step 3) and the cleaning of the WLASL clips (step 7) don't need to be
repeated: their outcomes, the final 67 words (`02_vocabulary_overlap/class_list.txt`) and the
clips removed (`08_wlasl_clean/excluded_clips.json`), are in the repository, and both steps can be
skipped.

Commands are given from the repository root.

## Requirements

- Python 3.11, with `pip install -r requirements.txt` from the repository root.
- `ffmpeg`, used by WLASL's preprocessing script to convert and trim the videos.
- `git` and `git-lfs`, for the Hugging Face copy of WLASL.
- About 15 GB of free disk space. The Hugging Face copy of WLASL is about 5 GB of videos, and git
  keeps a second copy of them. ASL Citizen isn't downloaded in full: only the 2,050 clips needed are
  fetched.

## Step 1: download WLASL100 (stage 01, `01_wlasl_download/`)

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
   of WLASL:
   ```
   git clone https://huggingface.co/datasets/Voxel51/WLASL hf_raw_videos
   python scripts/extract_asl100.py       # copies only the WLASL100 clips not already in videos/
   cd ../..
   ```

**Result:** `01_wlasl_download/videos/`, with 1,265 of the 2,038 WLASL100 clips. Links keep breaking,
so a fresh download may get fewer (see [Known differences](#known-differences-from-a-fresh-run)).

## Step 2: get ASL Citizen's split files

1. Read and accept the licence on the Microsoft Download Center page for ASL Citizen,
   <https://www.microsoft.com/en-us/download/details.aspx?id=105253> (project page:
   <https://www.microsoft.com/en-us/research/project/asl-citizen/>). The whole dataset is a single
   file, `ASL_Citizen.zip`, listed there as 42.8 GB.
2. Copy the split files and licence out of the zip, without downloading the whole thing:
   ```
   python -c "from remotezip import RemoteZip; z = RemoteZip('https://download.microsoft.com/download/b/8/8/b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip'); [z.extract(f, 'data/asl_citizen') for f in ('ASL_Citizen/splits/train.csv', 'ASL_Citizen/splits/val.csv', 'ASL_Citizen/splits/test.csv', 'ASL_Citizen/use.txt')]"
   ```

**Result:** `data/asl_citizen/ASL_Citizen/splits/{train,val,test}.csv`, which list every ASL Citizen
clip with its sign, signer and ASL-LEX code. Steps 3, 5 and 6 read them. The videos come in step 5,
once the vocabulary is known.

## Step 3: choose the vocabulary (stage 02, `02_vocabulary_overlap/`)

This step compares the two datasets' vocabularies. Its outcome, `02_vocabulary_overlap/class_list.txt`
(the final 67 words), is in the repository, so **you can skip to step 4.** Run step 3 only to
repeat the comparison.

```
cd pipeline/02_vocabulary_overlap && python find_overlap.py && cd ../..
```

A WLASL100 word is kept only if ASL Citizen has exactly one sign for it. ASL Citizen signs are
matched to a word by name, with or without a version number (*bed* → BED1, BED2). A word is
dropped if the matches have more than one ASL-LEX code, because the two datasets might then show
different signs for the same word.

- **Kept: 67 words**, listed in [`data/README.md`](../data/README.md#final-vocabulary-67-signs).
- **Dropped, several ASL Citizen signs (32):** basketball, bed, bowling, can, candy, cheat, city,
  cool, corn, dark, deaf, decide, doctor, dog, drink, eat, fine, hat, how, jacket, kiss, letter,
  pizza, right, same, shirt, short, tall, walk, want, what, woman.
- **Dropped, not in ASL Citizen (1):** forget.

*short* has several ASL Citizen signs, named by meaning (SHORTPERSON, SHORTDISTANCE, SHORTWORD)
rather than numbered, so the name match doesn't find them and `vocabulary.json` lists it under
"not in ASL Citizen". It is dropped either way.

**Result:** `class_list.txt` (class id, ASL Citizen sign and WLASL word for each of the 67 classes)
and `vocabulary.json` (the kept words with their ASL-LEX codes, and the dropped words with the
reason).

## Step 4: keep the WLASL clips of the 67 words (stage 03, `03_wlasl_filter/`)

```
cd pipeline/03_wlasl_filter && python filter_wlasl.py && cd ../..
```

**Result:** `videos/` with the 843 WLASL clips of the 67 words (out of 1,265), and
`wlasl_clips.json`, which lists them by word with each clip's source website and signer.

## Step 5: download the ASL Citizen clips (stage 04, `04_asl_citizen_download/`)

```
cd pipeline/04_asl_citizen_download && python download_asl_citizen.py && cd ../..
```

The script reads each clip straight out of `ASL_Citizen.zip` with HTTP range requests, so only the
clips of the 67 signs are downloaded.

**Result:** `videos/` with 2,050 ASL Citizen clips.

## Step 6: build the uncleaned split (stage 05, `05_split/`)

```
cd pipeline/05_split && python build_split.py && cd ../..
```

The ASL Citizen clips keep ASL Citizen's own signer-independent split. The WLASL clips are test
only.

**Result:** `split_uncleaned.json`: ASL Citizen train 973 / val 258 / test 819, WLASL 843.

## Step 7: clean the WLASL test clips (stages 06–08)

This step decides which WLASL clips to remove. Its outcome, `08_wlasl_clean/excluded_clips.json`,
is in the repository, so **to rebuild the dataset you can skip to step 8.** Run step 7 only to
repeat the review.

```
cd pipeline/06_pose_extraction
python pose_extraction.py --videos_dir ../03_wlasl_filter/videos --out_dir poses_wlasl
python pose_extraction_parallel.py --videos_dir ../04_asl_citizen_download/videos --out_dir poses_asl_citizen
cd ../07_baseline    && python train_baseline.py      # three baseline models, trained on ASL Citizen only
cd ../08_wlasl_clean && python first_round_eval.py    # then check the flagged clips by eye
cd ../..
```

Stage 06 extracts a MediaPipe Holistic pose file for every clip. Stage 07 trains three baseline
models (seeds 0–2) on ASL Citizen, written to `experiments/checkpoints/baseline_s{0,1,2}/`. Stage 08
uses them as follows.

Accuracy is only used to decide which clips to look at. Whether a clip is removed is decided by
watching it.

1. **First-round evaluation** (`first_round_eval.py`): the three baseline models classify all 843
   WLASL clips. Accuracy is reported per source website and per sign, along with the share of frames
   in which no person is detected.
2. **By source:** clips from northtexas get 3.2% top-1, compared with at least 30.1% for every other
   source with at least 20 clips, and no person is detected in 30.6% of their frames, against at
   most 1.2% for any other source. They were cut from compilation videos, and
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
words outside the 67, so they're never in the dataset. Step 8 reads this list from
`08_wlasl_clean/excluded_clips.json`, which holds the same clip ids and reasons.

## Step 8: assemble the final dataset (stage 09, `09_build_dataset/`)

```
cd pipeline/09_build_dataset && python build_final_dataset.py && cd ../..
```

This takes the uncleaned split from step 6, removes the clips in `excluded_clips.json`, and copies
the kept clips into `data/`. It ends by comparing the WLASL clips with
[`data/wlasl_test_clips.csv`](../data/wlasl_test_clips.csv) and names any that differ, for example
clips a fresh download couldn't retrieve.

**Result:** `data/split.json` (ASL Citizen 973 / 258 / 819, WLASL 791), `data/class_list.txt`,
`data/wlasl/` and `data/asl_citizen/videos/`.

## Step 9: extract the final poses

```
cd pipeline/06_pose_extraction
python pose_extraction.py --videos_dir ../../data/wlasl/videos --out_dir ../../data/poses/wlasl
python pose_extraction_parallel.py --videos_dir ../../data/asl_citizen/videos --out_dir ../../data/poses/asl_citizen
cd ../..
```

Same scripts and settings as stage 06, run on the final clips only. Each pose file holds the
MediaPipe Holistic body (33 points) and both hands (21 points each), extracted with
`model_complexity=2`.

**Result:** `data/poses/wlasl/` and `data/poses/asl_citizen/`. These are what `experiments/` reads.

## How the stages connect

| Stage | Folder | Reads | Writes |
|---|---|---|---|
| 01 | `01_wlasl_download/` | WLASL's index and videos | `metadata/WLASL_asl100.json`, `videos/` (1,265 clips) |
| — | `data/asl_citizen/` | ASL Citizen's zip | `ASL_Citizen/splits/{train,val,test}.csv` |
| 02 | `02_vocabulary_overlap/` | 01 index, ASL Citizen splits | `class_list.txt`, `vocabulary.json` |
| 03 | `03_wlasl_filter/` | 01 index and videos, 02 class list | `videos/` (843 clips), `wlasl_clips.json` |
| 04 | `04_asl_citizen_download/` | 02 class list, ASL Citizen splits | `videos/` (2,050 clips) |
| 05 | `05_split/` | 02 class list, 03 clip list, ASL Citizen splits | `split_uncleaned.json` |
| 06 | `06_pose_extraction/` | 03 and 04 videos | `poses_wlasl/`, `poses_asl_citizen/` |
| 07 | `07_baseline/` | 02 class list, 05 split, 06 ASL Citizen poses | `experiments/checkpoints/baseline_s{0,1,2}/` |
| 08 | `08_wlasl_clean/` | 03 clip list, 05 split, 06 WLASL poses, 07 models | `first_round_*.json`; with the check by eye, `excluded_clips.json` |
| 09 | `09_build_dataset/` | 02 class list, 03 and 04 videos, 03 clip list, 05 split, 08 `excluded_clips.json` | `data/split.json`, `data/class_list.txt`, `data/wlasl/`, `data/asl_citizen/videos/` |
| final poses | `06_pose_extraction/` | `data/` videos | `data/poses/` |

Poses are extracted twice. Stage 06 covers all 843 WLASL clips, because stages 07 and 08 need them
to decide which clips to remove. Step 9 covers only the final clips.

## Known differences from a fresh run

- **aslsignbank:** two of the 1,265 clips (65225 *book*, 65717 *fine*) came from aslsignbank. Its
  certificate doesn't match its address, so WLASL's downloader rejects it, and the clips aren't on
  Hugging Face. They were downloaded once with certificate checks turned off for that site. Without
  them a fresh run gets 1,264 clips. 65225 is a final test clip, so the WLASL test set would have
  790 clips.
