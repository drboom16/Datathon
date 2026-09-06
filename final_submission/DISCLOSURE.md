# Required disclosure

## External datasets

None. The official `train.csv` and `test(1).csv` are the only modelling inputs.

## External code, notebooks, repositories, public solutions

- LightGBM (`lightgbm.LGBMRegressor`), pandas, and numpy.
- No public competition notebook was copied as the final pipeline.

## Pretrained models

None. Every LightGBM model is trained from scratch on `train.csv`.

## AI tools / coding agents

Cursor (an AI coding assistant) was used to write and edit the training scripts, feature code, and this documentation. Model choices and the 50/50 blend were selected from the chronological winter holdouts described in `METHODOLOGY.md`.

## Manual modification or post-processing of predictions

- Predictions clipped at 0.
- Station-wise forward/backward fill on hours where PM10 is missing at all 12 stations.
- No hand-edits of individual rows.

## Information beyond the competition-provided files

None. Features, training labels, and predictions use only the official competition files.
