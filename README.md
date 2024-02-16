# Cham-GPT
LLM Chatbot for talking to a AI version of me

## Introduction

Cham-GPT is a LLM Chatbot for talking to a AI version of me. It is character-based and trained on small batches of text. While some of the sentences it generates are grammatical, most do not make sense, as the model does not learn the meaning of words. However, it can produce coherent structures resembling a play, with blocks of text generally beginning with a speaker's name.

## Setup

### Import TensorFlow and other libraries

```python
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os
import time
