# Expectations

| Model | Expected test accuracy |
|---|---|
| Word embeddings | 51–55% |
| Word embeddings + RNN | 53–57% |
| Transformer, randomly initialized | 55-58%|
| Transformer, pretrained | 70-73% |
| LLM | 87-92% |


## Word embeddings (word2vec)

I assume the model will be a bit better than random guessing, but not by much.
Averaging the word vectors loses the word order, and the two hypotheses often differ in only one or two words, so their averaged vectors end up very similar.
On top of that, the hypotheses often share vocabulary with the observations (lexical overlap), so the wrong hypothesis can look just as fitting as the correct one.

## Word embeddings + RNN

I expect the RNN to perform a bit better than word2vec, because it reads the words in order and can pick up things like negation ("He was never lost").
But I don't think it will get much better, because many examples need commonsense knowledge that neither model has.
For example:

- O1: After the match was called, the soccer team went to the locker room.
- O2: The food seemed to brighten up their moods and raise morale.
- h1: The coach bought dinner for his happy football players.
- h2: The coach bought dinner for his saddened players. ✓

To solve this you need to understand that "brighten up their moods" only makes sense if the players were sad before.
Word2vec doesn't help much here because antonyms like "happy" and "sad" get similar vectors (they appear in similar contexts).
Other examples need world knowledge (expl: spending "seconds" vs. "hours" at the gym before being exhausted), and word order doesn't help with that either.
Some examples might actually be easier, when a hypothesis shares the topic with the observations (expl: "spaceship" and "Kennedy Space Center").

I also wouldn't expect more than ~60%: in the ART paper (Bhagavatula et al., 2020), even a fine-tuned BERT only reaches about 69%.
An RNN with word2vec trained on relative small number of examples shouldn't beat that.
(Due to a different split and duplicate handling the results are not directly comaparable but still hold for a orientation)

Date written: 2026-10-25

## Transformer (randomly initialized)

I assume that this Transformer will not improve much over the previous models.
Since it is randomly initialized, it starts without any prior knowledge (not even pretrained word embeddings like the word2vec models in part 1), so it may not be able to capture the deeper contextual relations between the observations and the hypotheses.
The training set is also quite small, as already noted in part 1, which makes it hard to learn this from scratch.
I also expect the gap between validation and test accuracy from part 1 to persist, since it appeared for all models there.
Due to self-attention, the model can directly compare every hypothesis word with every observation word, so it might still achieve slightly better scores than the word2vec classifier or the RNN.

## Transformer (pretrained)

For this model I expect a bigger improvement. Here we can measure the impact of pretraining: the model has already been trained on large amounts of text (filling in masked words).
This gives it contextual representations, meaning the representation of a word depends on the sentence around it.
For example, it can recognize that "brighten up their moods" implies that the players were sad before, which is exactly the knowledge the soccer example needs.
In the ART paper, a fine-tuned BERT reaches about 69%. Since I use a more recent model with an improved pretraining recipe, I expect slightly better results, above 70%.

## LLM

For the last task I expect very good results, around 90%.
Human accuracy on ART is about 91%, and I think current LLMs can reach a similar level.
At first I expected even better results, but an open model of this size is probably less accurate than large closed models such as Claude. But out of interest, I will try to test both.
It is also important to keep in mind that the model may already know the dataset: it is public and well known, so it may have been part of the training data, and the model could partly remember the answers instead of reasoning about them.

Date written: 2026-11-26