# **How to Run Code**
---
All of the requirements for the project is inside **requirements.txt**. For the code to run properly, pip install them beforehand in a python venv

0) Clone the repository
```
git clone https://github.com/solyotamas/handsign_task.git
cd handsign_task
```
1) Create a python venv and install requirements
```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2) Place the original dataset *asl-signs folder itself (with the contents inside)* inside the root folder
3) Run the preprocessing scripts inside **preprocess_before.ipynb** Under the header 
"**2nd attempt 25 signs and all of their sequences**" 
4) Run the main pipeline

The main pipeline is inside **train_eval_pipeline.ipynb** 
notebook with runnable cells

This consist of:

    - training preparation
    - training of the basic model
    - training of the challenger model
    - evaluation & reflection of both

# **Approaches & Workflow**
---

1) Explored the given data in **exploration/feature_exploration.ipynb** notebook just to explore the structure of the given data, what features are available, how does one sequence look like

2) Explored preprocessing opportunities in **exploration/preprocessing_exploration.ipynb** notebook, missing landmarks, and generally what needed to be done to transform the given sequence data into a better / cleaner one

3) Wrote functions for these in **preprocessing/preprocessing_functions.py**, and tested these on single sequences in **preprocessing/test_preprocessing_functions.ipynb**

4) Created the dataset class and simple model classes, and wrote a training, evaluation function for the main pipeline, these are inside **models /** (probably not the most optimal structure, but kept them lightweight)

5) Firstly I tried to train the models while doing the preprocessing on the fly inside the dataset class but one batch took like 20 seconds to load so decided to preprocess and save the sequences I wanted to use beforehand (**preprocess_before.ipynb**). In The preprocessed sequences and their metadata are inside **dataset /**. With preprocessed sequences, it uses a little bit more disk space but the whole training is around 10 minutes for both models

6) The whole training / model evaluation pipeline is in **train_eval_pipeline.ipynb** notebook


