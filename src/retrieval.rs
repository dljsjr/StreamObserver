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

/// Split `text` into ~`chunk_words`-word windows — the retrievable passages, shared by the BM25 and
/// semantic indexes. (Fixed windows, not paragraphs: simplest unit that still localizes a hit.)
pub fn chunk(text: &str, chunk_words: usize) -> Vec<String> {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .chunks(chunk_words.max(1))
        .map(|w| w.join(" "))
        .collect()
}

/// Build a BM25 index over `text`, chunked into ~`chunk_words`-word windows.
pub fn index(text: &str, chunk_words: usize) -> Corpus {
    index_chunks(chunk(text, chunk_words))
}

/// Build a BM25 index over pre-chunked passages — so a hybrid index can share the EXACT same chunks
/// (and chunk indices) as a `SemanticIndex`, which RRF fusion requires.
pub fn index_chunks(chunks: Vec<String>) -> Corpus {
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

impl Corpus {
    /// The verbatim text of chunk `i` (for returning a fused/ranked hit by index).
    pub fn chunk_text(&self, i: usize) -> &str {
        &self.chunks[i]
    }
}

/// BM25 ranking: chunk indices with a positive score, highest first. Chunks with no query-term
/// overlap are omitted (BM25 doesn't rank them). The basis for both `search` and RRF fusion.
pub fn rank_bm25(corpus: &Corpus, query: &str) -> Vec<usize> {
    let n = corpus.chunks.len() as f32;
    let q = terms(query);
    let mut scored: Vec<(usize, f32)> = corpus
        .tf
        .iter()
        .enumerate()
        .filter_map(|(i, tf)| {
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
            (score > 0.0).then_some((i, score))
        })
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));
    scored.into_iter().map(|(i, _)| i).collect()
}

/// Top BM25 chunk for `query`, or `None` if nothing overlaps.
pub fn search(corpus: &Corpus, query: &str) -> Option<String> {
    rank_bm25(corpus, query)
        .first()
        .map(|&i| corpus.chunks[i].clone())
}

/// A semantic index: per-chunk L2-normalized embeddings, aligned by index with a `Corpus`'s chunks
/// (so RRF can fuse the two by index, and the hit text comes from `Corpus::chunk_text`). The
/// embedding itself is the impure part (done by the caller); this + `rank_semantic` are pure cosine.
pub struct SemanticIndex {
    embeddings: Vec<Vec<f32>>,
}

impl SemanticIndex {
    /// `embeddings[i]` describes chunk `i` (L2-normalized), aligned with the paired `Corpus`.
    pub fn new(embeddings: Vec<Vec<f32>>) -> Self {
        Self { embeddings }
    }
}

/// Semantic ranking: ALL chunk indices, highest cosine first (every chunk has a similarity — no
/// term-overlap gate, unlike BM25). Cosine == dot product since both sides are L2-normalized.
pub fn rank_semantic(index: &SemanticIndex, query: &[f32]) -> Vec<usize> {
    let mut scored: Vec<(usize, f32)> = index
        .embeddings
        .iter()
        .enumerate()
        .map(|(i, e)| (i, dot(e, query)))
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));
    scored.into_iter().map(|(i, _)| i).collect()
}

/// Reciprocal Rank Fusion (Cormack et al. 2009): `score(doc) = Σ_list 1/(k + rank)` with 1-based
/// rank; `k = 60` is the canonical default. Fuses heterogeneous rankings (BM25 + semantic) robustly
/// — it uses only ranks, so no score normalization across the two very different scales is needed.
/// Returns the highest-fused chunk index across `rankings`, or `None` if all are empty.
pub fn rrf_best(rankings: &[&[usize]], k: f32) -> Option<usize> {
    let mut scores: HashMap<usize, f32> = HashMap::new();
    for list in rankings {
        for (rank0, &doc) in list.iter().enumerate() {
            *scores.entry(doc).or_insert(0.0) += 1.0 / (k + rank0 as f32 + 1.0);
        }
    }
    scores
        .into_iter()
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(doc, _)| doc)
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
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

    #[test]
    fn semantic_ranking_orders_by_cosine() {
        // Three orthogonal unit vectors; the query points nearest index 1, then 0, then 2.
        let idx = SemanticIndex::new(vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]]);
        assert_eq!(rank_semantic(&idx, &[0.3, 0.9, 0.0]), vec![1, 0, 2]);
    }

    #[test]
    fn semantic_empty_index_ranks_nothing() {
        let idx = SemanticIndex::new(vec![]);
        assert!(rank_semantic(&idx, &[1.0, 0.0]).is_empty());
    }

    #[test]
    fn rrf_rewards_agreement_across_rankings() {
        // doc 2 is top of one list and second of the other → best fused; doc 0/1 each top one list.
        let bm25 = [0usize, 2, 1];
        let semantic = [2usize, 1, 0];
        assert_eq!(rrf_best(&[&bm25, &semantic], 60.0), Some(2));
    }

    #[test]
    fn rrf_includes_docs_present_in_only_one_ranking() {
        // BM25 found nothing (empty); semantic still ranks → fusion falls back to semantic's top.
        let bm25: [usize; 0] = [];
        let semantic = [5usize, 3, 9];
        assert_eq!(rrf_best(&[&bm25, &semantic], 60.0), Some(5));
        assert_eq!(rrf_best(&[&[], &[]], 60.0), None);
    }
}
