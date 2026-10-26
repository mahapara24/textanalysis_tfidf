# Lab 15 Answers

## Challenge 1: Conversational Filler & Numeric Noise

The TF-IDF formula is TFIDF = TF * log(N / DF). Even if a word like "know" shows up in 25% of posts, IDF is still log(1/0.25) = log(4) which is about 1.39, so it is not rare enough to get pushed down to basically zero. Normal NLTK stopwords also don't include filler words like know/like/just/think, so they still get weight if they appear a lot in a post.

After I added token_pattern so we only keep letters with length >= 3, set max_df=0.5, and added those custom stopwords, the top cluster words looked more useful (cars, mac/hardware, medical stuff) instead of random chat words.

## Challenge 2: Hardware Vocabulary Overlap

From the crosstab of true_label vs cluster, sci.electronics and comp.sys.mac.hardware overlapped the most. Looking at some misclustered posts, shared words that kept showing up were things like board, chip, drive, power, and memory — basically hardware terms that both groups use.

Using ngram_range=(1, 2) helped a bit because bigrams can pick up pairs like "hard drive" or mac-related phrases, which gives more context than single words alone.

## Challenge 3: Rant Length Bias (BoW vs. TF-IDF)

I compared the L2 norms of the longest and shortest posts. For Bag of Words the long post had a much bigger norm than the short one. For TF-IDF the norms were closer (around 1 because of L2 normalization).

That is why KMeans struggles on raw BoW — long rants dominate the distance just because they have more word counts. TfidfVectorizer normalizes each document, so length doesn't mess up the clustering as much.

## Challenge 4: Regulated Compliance Explainability

I implemented explain_document so for doc 412 it prints the top 5 TF-IDF terms with their weights. That way you can literally see which words caused the document to land in its cluster.

TF-IDF is better than a black-box LLM here because the weights are clear and you can recalculate them. An auditor can check the exact features instead of trusting a model that doesn't explain itself. For regulated healthcare routing that kind of transparency matters a lot.
