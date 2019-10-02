# fm_series

### Abstruct
Use tensorflow to write fm / ffm / deepfm(comming soon) /...

### structure:
- lib/fm.py -> fm model written by python normal
- lib/tensorflow_fm.py -> fm model written by tensorflow
- lib/tensorflow_ffm.py -> ffm model written by tensorflow

- notebookfile/fm_fit_func_and_classification.ipynb -> run fm.py
- notebookfile/tensorflow_fm.ipynb -> run tensorflow_fm.py
- notebookfile/tensorflow_ffm.ipynb -> run tensorflow_ffm.py


### How to run

- `cd fm_series`
- `pipenv install`
- `pipenv run jupyter lab --no-browser`
- In pycharm 
    - Click `Run`
    - Type url and then the notebook file can run.


