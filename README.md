# RODALIES DATASET: A DATASET FOR PIANO SCORE DIFFICULTY ANALYSIS

## Abstract
Information on the performance difficulty of a music score is fundamental for structuring and exploring score 
collections. Despite its importance for music education, performance difficulty prediction is not yet solved mainly 
due to the lack of adequate data. In this paper, we present the Rodalies dataset, a machine-readable piano score dataset
with difficulty annotations obtained from the renowned  classical music publisher Henle Verlag. We create the dataset 
by matching metadata between public domain scores available from MuseScore with Henle Verlag’s difficulty labels. Then, 
an expert pianist reviewed and corrected the automatic matching. The resulting Rodalies dataset contains 551 classical
piano pieces, spanning 9 difficulty levels and 29 composers ranging from baroque to 20th Century. 
Accompanying the dataset, we compute different statistical features of the scores with a proven relationship to music performance complexity and a baseline difficulty analysis model based on gated recurrent units with attention for 
difficulty analysis. We discuss different strategies to handle unbalanced piece lengths and the use of feature 
representations extracted from two related tasks: piano fingering and expressive performance modelling. We report 41.44% 
balanced accuracy and 2.35 median square error on the proposed 9 levels of difficulty for the best performing model.

## Project Structure
`run_model_full_bootstrap.py`: train the DeepGRU model with bootstrap training methodology for the 4 representations.

`run_model_full_without.py`: train the DeepGRU model with bootstrap training methodology for the 4 representations.

`old_school_experiment.ipynb`: the *GNBold_school* experiment with the feature engineering `old_school_features.json` 

`load_representations.py`,  `data.py` and `utils.py`: util functions.

Directory `Fingers` contains the fingers precomputed for Rodalies and Mikrokosmos-difficulty dataset.

Directory `representations` contains the processed features to handle them easily and fast from `data.py`.

Directories `Rodalies_virtuoso_embedding` and `Mikrokosmos-difficulty_virtuoso_embedding` contain the virtuoso embeddings.

`tables.ipynb`: tables showing the results of the paper.

## Train
To re-train the models execute: `python3 run_model_full_bootstrap.py [log_path] [representation]` with `log_path` the path to the tensorboard log directory and `representation`: `note`, `pp`, `prob` or `virtuoso`.

## Pre-trained models
It will be included if the paper is accepted. 