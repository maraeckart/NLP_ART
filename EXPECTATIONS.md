# Expectations

| Model | Expected test accuracy |
|---|---|
| Word embeddings | 51–55% |
| Word embeddings + RNN | 53–57% |
| Transformer, randomly initialized | _TODO_ |
| Transformer, pretrained | _TODO_ |
| LLM | _TODO_ |

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