RNN Language Model Pytorch
===

Pytorch implementation of RNN language model for text generation task.


## Environment
- Python 3.6
- Pytorch 1.1.0
- Debugsummary 0.1
- Numpy 1.18.1


## Dataset
We use ROCStories, see [A Corpus and Cloze Evaluation for Deeper Understanding of Commonsense Stories](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/06/short-commonsense-stories.pdf).


## Training RNN Language Model
You can use either rnn cell, gru cell or lstm cell. Change the configurations in the code.
```bash
python run_lm.py
```


## References
This code is based on dennybritz's [cnn-text-classification-tf](https://github.com/dennybritz/cnn-text-classification-tf).
Many thanks!
