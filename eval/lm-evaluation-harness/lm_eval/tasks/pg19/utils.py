import re


def process_results(doc, results):
    (loglikelihood,) = results
    # PG19 word count calculation
    _words = len(re.split(r"\s+", doc["text"]))
    _bytes = len(doc["text"].encode("utf-8"))
    return {
        "word_perplexity": (loglikelihood, _words),
        "byte_perplexity": (loglikelihood, _bytes),
        "bits_per_byte": (loglikelihood, _bytes),
    }
