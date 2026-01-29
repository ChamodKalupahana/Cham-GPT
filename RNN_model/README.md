# Cham-GPT (Early 2024)
LLM Chatbot for talking to a AI version of me

## Introduction

Cham-GPT is a LLM Chatbot for talking to a AI version of me. It is character-based and trained on small batches of text. While some of the sentences it generates are grammatical, most do not make sense, as the model does not learn the meaning of words. However, it can produce coherent structures resembling a conversation, with blocks of text generally beginning with my name.

This is an ongoing project that I hope to slowly build over time into a chatbot running on the cloud so that anyone can access it through a web server.

## Setup

### Import TensorFlow and other libraries

```python
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np