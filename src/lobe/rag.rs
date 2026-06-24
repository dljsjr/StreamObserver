//! #8 — native tool-calling RAG, the most self-contained interjection concern: the result types, the
//! `Lobe::rag` generation pass, and the lenient parser for gemma-4's native tool-call output. The
//! retrieval BACKEND is the `run_retrieval` stub in `main`; this module only decides + parses. The
//! prompt itself lives in `crate::prompt` (the gemma-4 chat-format owner).

use super::{argmax, word_aligned, Lobe, GEN_SEQ};
use crate::backend::{Backend, Session};
use anyhow::Result;

/// Where a retrieval directive (#8) points: this session's own memory, or the external KB.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Source {
    Mem,
    Rag,
}

/// A parsed native tool call: the observer wants to look `query` up in `source`.
#[derive(Clone, Debug)]
pub struct RetrievalDirective {
    pub source: Source,
    pub query: String,
}

/// Result of a #8 RAG pass: the free thinking before the call, the optional retrieval directive the
/// model emitted as a native tool call, the snippet retrieval returned (for the `retrieval` event),
/// and the grounded reply the model generated AFTER the snippet was fed back. `directive`/`retrieved`/
/// `response` are None when the model abstained or retrieval found nothing.
#[derive(Clone, Debug, Default)]
pub struct RagOutcome {
    pub thought: String,
    pub directive: Option<RetrievalDirective>,
    pub retrieved: Option<String>,
    pub response: Option<String>,
}

impl<B: Backend> Lobe<'_, B> {
    /// #8 — native tool-calling RAG hook. Declares a `search` tool in gemma-4's native format (see
    /// `prompt::rag_prompt`), lets the observer think and (if warranted) emit a native tool call, and
    /// parses the result into a free-text thought (the interjection) + an optional retrieval directive.
    /// No grammar — the modal think→call structure is the model's own. Generation renders special
    /// tokens as text (`detok`, not `detok_gen`) so the `<|tool_call>…<tool_call|>` markers survive for
    /// parsing. Snippet-based (uses the recent window); stops at the model's turn-close after the call.
    pub fn rag<F: FnMut(Source, &str) -> Option<String>>(
        &mut self,
        surprising: &str,
        max: usize,
        mut retrieve: F,
    ) -> Result<RagOutcome> {
        self.session.clear_seq(GEN_SEQ as u32)?;
        let recent: String = self.recent.iter().map(String::as_str).collect();
        let prompt = crate::prompt::rag_prompt(word_aligned(&recent).trim(), surprising);
        let t0 = std::time::Instant::now();
        let toks = self.tokenize(&prompt, true)?;
        let prompt_len = toks.len();
        let mut logits = self.decode_seq(&toks, 0, GEN_SEQ)?;
        let mut raw = String::new();
        let mut pos = toks.len() as i32;
        let mut produced = 0usize;
        // PHASE 1 — think + (maybe) call. Stop at the call-close marker so we can answer it inline,
        // or at a turn/eog boundary (abstain → just the sentence), or the cap.
        for _ in 0..max {
            let tok = argmax(&logits);
            if self.engine.is_eog(tok) || Some(tok) == self.eot {
                break;
            }
            raw.push_str(&self.detok(tok)); // specials-as-text so the tool-call markers survive
            logits = self.decode_seq(&[tok], pos, GEN_SEQ)?;
            pos += 1;
            produced += 1;
            if raw.contains("<tool_call|>") {
                break; // the call is complete — go answer it
            }
        }
        let mut outcome = parse_rag_output(&raw);
        // PHASE 2 — feedback loop. If a call was emitted and retrieval hits, feed the result back
        // INLINE in the same model turn (gemma-native `<|tool_response>`) and continue generating a
        // grounded reply. detok_gen (specials suppressed) — this part is the user-facing aside.
        if let Some(d) = &outcome.directive {
            if let Some(snippet) = retrieve(d.source, &d.query) {
                let bridge = crate::prompt::rag_tool_response(&snippet);
                let btoks = self.tokenize(&bridge, false)?; // continuation, no BOS
                logits = self.decode_seq(&btoks, pos, GEN_SEQ)?;
                pos += btoks.len() as i32;
                let mut resp = String::new();
                for _ in 0..max {
                    let tok = argmax(&logits);
                    if self.engine.is_eog(tok) || Some(tok) == self.eot {
                        break;
                    }
                    resp.push_str(&self.detok_gen(tok));
                    logits = self.decode_seq(&[tok], pos, GEN_SEQ)?;
                    pos += 1;
                    produced += 1;
                }
                outcome.retrieved = Some(snippet);
                outcome.response = Some(resp.trim().to_string());
            }
        }
        self.session.clear_seq(GEN_SEQ as u32)?;
        let (src, query) = match &outcome.directive {
            Some(d) => (
                match d.source {
                    Source::Mem => "mem",
                    Source::Rag => "rag",
                },
                d.query.as_str(),
            ),
            None => ("none", ""),
        };
        tracing::info!(
            target: "lobe::rag", kind = "rag",
            stream_index = self.stream_index as u64, trigger_token = %surprising,
            prompt_tokens = prompt_len as u64, produced = produced as u64,
            latency_us = t0.elapsed().as_micros() as u64,
            directive_source = src, directive_query = %query,
            thought = %outcome.thought, raw_output = %raw,
            retrieved = outcome.retrieved.as_deref().unwrap_or(""),
            response = outcome.response.as_deref().unwrap_or(""),
            prompt = %prompt, // raw model input, verbatim
            "rag"
        );
        Ok(outcome)
    }
}

/// Parse a #8 RAG generation into a free-text thought + optional retrieval directive. The model's
/// output looks like `[free thought]<|tool_call>call:search{query:<|"|>…<|"|>,source:<|"|>…<|"|>}…`
/// (special tokens rendered as text). Everything before the call is the thought; the call args are
/// extracted leniently. No call → abstain (directive None).
fn parse_rag_output(raw: &str) -> RagOutcome {
    match raw.split_once("<|tool_call>") {
        Some((before, after)) => {
            let source = match extract_tool_arg(after, "source").as_deref() {
                Some(s) if s.trim().eq_ignore_ascii_case("rag") => Source::Rag,
                _ => Source::Mem, // default to session memory if unspecified/odd
            };
            let directive = extract_tool_arg(after, "query").map(|q| RetrievalDirective {
                source,
                query: q.trim().to_string(),
            });
            RagOutcome {
                thought: before.trim().to_string(),
                directive,
                ..Default::default()
            }
        }
        None => RagOutcome {
            thought: raw.trim().to_string(),
            ..Default::default()
        },
    }
}

/// Extract `{key}:<|"|>VALUE<|"|>` from a gemma-4 tool-call body (the `<|"|>` quote-token delimiter
/// is rendered as literal text by `detok`).
fn extract_tool_arg(s: &str, key: &str) -> Option<String> {
    let pat = format!("{key}:<|\"|>");
    let start = s.find(&pat)? + pat.len();
    let rest = &s[start..];
    let end = rest.find("<|\"|>")?;
    Some(rest[..end].to_string())
}
