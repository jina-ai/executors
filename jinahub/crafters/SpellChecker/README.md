# SpellChecker

A simple [BKTree](https://en.wikipedia.org/wiki/BK-tree)-based spelling corrector.

## Basic usage

Requires a training dataset of sentences.

```python
input_training_data = [
        'they can go quite fast',
        'there were the new Japanese Honda',
    ]

train_docs = DocumentArray([
    Document(content=t) for t in input_training_data
    ])

with Flow().add(uses=SpellChecker) as f:
    f.post(on='/train', inputs=train_docs)

```

Then the spelling of your text Documents can be fixed as follows:

```python
...
input_docs = DocumentArray(
            [Document(content=t) for t in incorrect_text]
            )
results = f.post(on='/index', inputs=input_docs, return_results=True)
print(results[0].docs)  # documents can be found here

```

Note that calling the `/train` again will delete the existing model.

<!-- version=v0.1 -> 