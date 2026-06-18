# Twitter Sentiment Analysis - NLP

This notebook trains a tweet classifier for the Analytics Vidhya style
hate-speech task. It reads `Twitter Sentiments.csv`, cleans tweet text, explores
hashtags and word clouds, vectorizes the cleaned tweets, and trains a machine
learning model.

## Dataset Format

The notebook expects a CSV file named `Twitter Sentiments.csv` with these
columns:

```csv
id,label,tweet
```

The `label` value is binary:

- `0`: not racist or sexist
- `1`: racist or sexist

## Run The Notebook

```bash
pip install pandas numpy matplotlib seaborn nltk wordcloud scikit-learn
jupyter notebook "Twitter Sentiment Analysis - NLP_Project_89_Twitter Sentiment Analysis - NLP.ipynb"
```

Keep `Twitter Sentiments.csv` in the repository root before running the loading
cell.

## Prepare TweetClaw Exports

Use `scripts/tweetclaw_to_hate_speech_csv.py` to convert labeled
[TweetClaw](https://github.com/Xquik-dev/tweetclaw) JSON, JSONL, NDJSON, or CSV
exports into the notebook's `id,label,tweet` format:

```bash
python scripts/tweetclaw_to_hate_speech_csv.py tweetclaw-labeled.json "Twitter Sentiments.csv"
```

The converter only accepts binary hate-speech labels. It preserves `0` and `1`
values, accepts boolean `true` and `false` values from `hate_speech` fields, and
also accepts clear text labels such as `not_hate`, `normal`, `hate`, `racist`,
`sexist`, `offensive`, or `toxic`. It rejects generic
sentiment labels such as `positive`, `negative`, and `neutral` because those do
not mean the same thing as this notebook's hate-speech target.

Do not include account credentials, cookies, session files, direct messages, or
private user data in shared exports.
