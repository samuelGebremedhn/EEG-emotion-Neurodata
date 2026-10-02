# Model card

## Intended use
Educational/research exploration of EEG-based binary valence or arousal classification on DEAP.

## Out-of-scope use
Clinical diagnosis, mental-health assessment, employment screening, surveillance, or claims about a person's true emotional state.

## Data
DEAP EEG recordings and self-reported affective ratings. Dataset access and redistribution are governed by the DEAP EULA.

## Evaluation
The primary evaluation is subject-independent. Random window splits are retained only as a leakage demonstration.

## Limitations
Small controlled dataset, subjective labels, participant variability, recording artifacts, stimulus-specific responses, and uncertain generalization outside the dataset.

## Ethical note
A classifier output is an estimate produced from a particular dataset and feature/model pipeline. It should not be interpreted as direct measurement of a person's internal emotional experience.
