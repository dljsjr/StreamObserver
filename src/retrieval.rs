//! Lexical retrieval over a text corpus — the real backend behind `run_retrieval` (#8). No model, no
//! embeddings: chunk the text, score chunks against a query with BM25, return the best snippet. This
//! is the "external knowledge base" the observer consults when a surprising entity warrants a lookup
//! (e.g. the novel being read). Functional core: `index` and `search` are pure functions over plain
//! data, so they test with literals. BM25 (not naive overlap) so common words like "the" don't win.

use std::collections::HashMap;

/// BM25 parameters (the standard defaults): term-frequency saturation and length normalization.
const K1: f32 = 1.5;
const B: f32 = 0.75;

/// An indexed corpus: the retrievable chunks plus the BM25 statistics over them. Built once by
/// `index`; queried by `search`.
pub struct Corpus {
    /// The retrievable snippets, verbatim (what `search` returns).
    chunks: Vec<String>,
    /// Per-chunk term frequencies (lowercased alphanumeric terms).
    tf: Vec<HashMap<String, u32>>,
    /// Document frequency: how many chunks contain each term.
    df: HashMap<String, u32>,
    /// Per-chunk length in terms, and the average — BM25's length normalization.
    lens: Vec<u32>,
    avgdl: f32,
}

/// Split into lowercased alphanumeric terms (≥ 2 chars), the unit both indexing and querying score on.
fn terms(s: &str) -> Vec<String> {
    s.split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() >= 2)
        .map(str::to_lowercase)
        .collect()
}

/// Build a BM25 index by chunking `text` into windows of ~`chunk_words` whitespace-separated words.
/// (Fixed windows, not paragraphs — simplest unit that still localizes a hit to a readable passage.)
pub fn index(text: &str, chunk_words: usize) -> Corpus {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut chunks = Vec::new();
    for w in words.chunks(chunk_words.max(1)) {
        chunks.push(w.join(" "));
    }
    let mut tf = Vec::with_capacity(chunks.len());
    let mut df: HashMap<String, u32> = HashMap::new();
    let mut lens = Vec::with_capacity(chunks.len());
    for chunk in &chunks {
        let mut counts: HashMap<String, u32> = HashMap::new();
        let ts = terms(chunk);
        lens.push(ts.len() as u32);
        for t in ts {
            *counts.entry(t).or_insert(0) += 1;
        }
        for term in counts.keys() {
            *df.entry(term.clone()).or_insert(0) += 1;
        }
        tf.push(counts);
    }
    let total: u64 = lens.iter().map(|&l| l as u64).sum();
    let avgdl = if chunks.is_empty() {
        0.0
    } else {
        total as f32 / chunks.len() as f32
    };
    Corpus { chunks, tf, df, lens, avgdl }
}

/// Return the highest-scoring chunk for `query` under BM25, or `None` if nothing scores above zero
/// (no query term occurs in the corpus) or the corpus is empty.
pub fn search(corpus: &Corpus, query: &str) -> Option<String> {
    if corpus.chunks.is_empty() {
        return None;
    }
    let n = corpus.chunks.len() as f32;
    let q = terms(query);
    let mut best: Option<(usize, f32)> = None;
    for (i, tf) in corpus.tf.iter().enumerate() {
        let dl = corpus.lens[i] as f32;
        let mut score = 0.0f32;
        for term in &q {
            let f = match tf.get(term) {
                Some(&f) => f as f32,
                None => continue,
            };
            let df = *corpus.df.get(term).unwrap_or(&0) as f32;
            // IDF with the standard BM25 +0.5 smoothing (downweights corpus-wide common terms).
            let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
            let denom = f + K1 * (1.0 - B + B * dl / corpus.avgdl);
            score += idf * (f * (K1 + 1.0)) / denom;
        }
        if score > 0.0 && best.is_none_or(|(_, b)| score > b) {
            best = Some((i, score));
        }
    }
    best.map(|(i, _)| corpus.chunks[i].clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_finds_the_chunk_with_the_query_term() {
        let text = "the cat sat on the mat. \
                    the whale breached beside the doomed ship Pequod. \
                    the dog ran across the green field.";
        let c = index(text, 6); // ~6-word chunks
        let hit = search(&c, "Pequod whale").expect("a chunk mentions the whale");
        assert!(hit.to_lowercase().contains("whale") || hit.to_lowercase().contains("pequod"));
    }

    #[test]
    fn common_words_do_not_dominate() {
        // "the" is in every chunk; a query of only stopword-ish common terms shouldn't strongly
        // prefer any one chunk, but a distinctive term should pull its chunk to the top.
        let text = "the the the apple. the the the banana. the the the cherry.";
        let c = index(text, 4);
        let hit = search(&c, "banana").unwrap();
        assert!(hit.contains("banana"));
    }

    #[test]
    fn no_matching_term_returns_none() {
        let c = index("alpha beta gamma delta", 2);
        assert!(search(&c, "zzz nonexistent").is_none());
    }

    #[test]
    fn empty_corpus_returns_none() {
        let c = index("", 8);
        assert!(search(&c, "anything").is_none());
    }
}
