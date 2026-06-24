//! gemma-4 chat-prompt construction — the single owner of "how we talk to gemma-4".
//!
//! Everything that emits gemma-4 chat-turn syntax lives here, so the turn tokens aren't hand-rolled
//! across `main.rs` and `lobe.rs` (the split that made the `<|turn>user`→`<|turn>system` fix a
//! multi-site change). These are PURE builders over primitives — no `Lobe` state, no config enums —
//! so they're trivially unit-testable; callers gather their state and the resolved options, then call
//! in here.
//!
//! ## gemma-4 turn tokens (verified against the GGUF `tokenizer.chat_template` + Google's docs)
//! - `<|turn>` opens a turn, `<turn|>` closes it; the role (`system`/`user`/`model`) follows the open.
//!   These are SINGLE special tokens — NOT gemma-2/3's `<start_of_turn>`/`<end_of_turn>` (absent from
//!   the vocab; using them tokenizes as literal text and the model never enters a turn).
//! - gemma-4 has a real `system` role (its own `<|turn>system…<turn|>` block, NOT folded into the
//!   user turn) — that's where the persona/sink goes.
//! - Native tools: `<|tool>declaration:…<tool|>` declares; `<|"|>` is the string delimiter in tool
//!   structured data; the model emits `<|tool_call>call:…<tool_call|>`.
//! - A generation prompt ends with an open `<|turn>model\n` (the model writes the turn body next).

/// The framed system preamble (the pinned StreamingLLM sink). With `frame`, the persona goes in
/// gemma-4's dedicated SYSTEM turn and the model turn is opened so the stream is scored as the
/// model's own reasoning; without it, the persona is pinned as plain text. (The caller tokenizes
/// this with `add_bos = true` — BOS is the canonical first attention sink.)
pub fn system_preamble(system_prompt: &str, frame: bool) -> String {
    if frame {
        format!("<|turn>system\n{system_prompt}<turn|>\n<|turn>model\n")
    } else {
        system_prompt.to_string()
    }
}

/// SNIPPET-mode interjection ask: a standalone prompt (the caller tokenizes with BOS) carrying the
/// `recent` window as text, since snippet mode can't fork the live KV. `recent` should already be
/// word-aligned/trimmed. NB phrasing is empirical — recent text FIRST, instruction after, or
/// gemma-4-E2B continues the recent text instead of reacting to it.
pub fn interject_prompt_snippet(recent: &str) -> String {
    format!(
        "<|turn>user\nHere is the passage the text has just reached:\n\n{recent}\n\nGive your \
         brief aside on it — what's happening, what it's doing, what it means.<turn|>\n\
         <|turn>model\n",
    )
}

/// CONTEXT-mode interjection ask: appended AFTER the forked full context (no BOS — it continues that
/// context). Closes the in-progress turn, opens a user turn spotlighting the delta `span` (+ optional
/// `novelty_block` = the model's recent asides) and a closing `novelty_clause`, then opens the model
/// turn. `continuous` picks the framing: false = "comment on this quoted passage" (Passage, control);
/// true = "pick up your running commentary" (Continuous). `span` should be word-aligned/trimmed and
/// `novelty_block`/`novelty_clause` pre-resolved by the caller (empty strings to omit).
pub fn interject_ask_context(
    span: &str,
    novelty_block: &str,
    novelty_clause: &str,
    continuous: bool,
) -> String {
    if continuous {
        format!(
            "<turn|>\n<|turn>user\nYou've been musing as you read, and the text has moved on \
             to:\n\n\u{201c}{span}\u{201d}{novelty_block}\n\nGo on — pick up your running commentary in \
             your own voice{novelty_clause}.<turn|>\n<|turn>model\n",
        )
    } else {
        format!(
            "<turn|>\n<|turn>user\nHere is the passage the text has just reached, quoted for you \
             to comment on (not continue):\n\n\u{201c}{span}\u{201d}{novelty_block}\n\nGive your aside \
             on it — what's happening here, what it's doing, what it means, and how it connects \
             to what came before{novelty_clause}.<turn|>\n<|turn>model\n",
        )
    }
}

/// The gemma-4 native `search` tool declaration (strings delimited by the `<|"|>` quote token), for
/// the #8 RAG hook's system turn.
pub const SEARCH_TOOL: &str = r#"<|tool>declaration:search{description:<|"|>Look up an unfamiliar term, name, place, or reference.<|"|>,parameters:{properties:{query:{description:<|"|>what to look up<|"|>,type:<|"|>STRING<|"|>},source:{description:<|"|>mem for this session's earlier context, rag for the external knowledge base<|"|>,type:<|"|>STRING<|"|>}},required:[<|"|>query<|"|>,<|"|>source<|"|>],type:<|"|>OBJECT<|"|>}}<tool|>"#;

/// The #8 RAG prompt: a system turn declaring the `search` tool + a user turn with the `recent`
/// train-of-thought, asking the model to think and (if warranted) call the tool. `recent` should be
/// word-aligned/trimmed. The caller tokenizes with BOS and parses the output for a `<|tool_call>`.
pub fn rag_prompt(recent: &str, surprising: &str) -> String {
    format!(
        "<|turn>system\n{SEARCH_TOOL}\n<turn|>\n<|turn>user\nHere is the train of thought you've \
         been following:\n\n{recent}\n\nThe token \"{surprising}\" stood out. If it is an \
         unfamiliar term, name, place, or reference worth looking up, call the search tool. \
         Otherwise just say in a sentence what you noticed.<turn|>\n<|turn>model\n",
    )
}

/// The #8 tool-response block, fed back INLINE in the same model turn right after the model's
/// `<tool_call|>` so generation continues grounded in the result. The format is gemma-4's own
/// (derived from the GGUF `tokenizer.chat_template` `format_tool_response_block` macro, string
/// response branch): `<|tool_response>response:search{value:<|"|>…<|"|>}<tool_response|>`. Tokenize
/// with `add_bos = false` — it's a continuation, not a new sequence.
pub fn rag_tool_response(snippet: &str) -> String {
    format!("<|tool_response>response:search{{value:<|\"|>{snippet}<|\"|>}}<tool_response|>")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn framed_preamble_uses_the_system_role_and_opens_the_model_turn() {
        let p = system_preamble("PERSONA", true);
        assert_eq!(p, "<|turn>system\nPERSONA<turn|>\n<|turn>model\n");
        // gemma-2/3's markers must NOT appear, and the persona is NOT in a user turn.
        assert!(!p.contains("<start_of_turn>"));
        assert!(!p.contains("<|turn>user"));
    }

    #[test]
    fn unframed_preamble_is_plain_text() {
        assert_eq!(system_preamble("PERSONA", false), "PERSONA");
    }

    #[test]
    fn context_ask_closes_then_opens_user_and_model_turns() {
        let a = interject_ask_context("SPAN", "", "", false);
        assert!(a.starts_with("<turn|>\n<|turn>user\n")); // closes the open model turn, opens user
        assert!(a.ends_with("<|turn>model\n")); // opens the model turn for the aside
        assert!(a.contains("\u{201c}SPAN\u{201d}")); // span is curly-quoted
        // Passage framing, not Continuous.
        assert!(a.contains("comment on (not continue)"));
        assert!(interject_ask_context("S", "", "", true).contains("pick up your running commentary"));
    }

    #[test]
    fn context_ask_injects_novelty_block_and_clause() {
        let a = interject_ask_context("S", "\n\nYou recently said:\n- foo", " — fresh angle", false);
        assert!(a.contains("You recently said:\n- foo"));
        assert!(a.contains(" — fresh angle"));
    }

    #[test]
    fn snippet_prompt_puts_recent_first_then_instruction() {
        let p = interject_prompt_snippet("RECENT");
        let recent_at = p.find("RECENT").unwrap();
        let ask_at = p.find("Give your").unwrap();
        assert!(recent_at < ask_at, "recent text must precede the instruction");
        assert!(p.ends_with("<|turn>model\n"));
    }

    #[test]
    fn rag_prompt_declares_the_tool_in_a_system_turn() {
        let p = rag_prompt("RECENT", "tok");
        assert!(p.starts_with("<|turn>system\n<|tool>declaration:search"));
        assert!(p.contains("<|turn>user\n"));
        assert!(p.ends_with("<|turn>model\n"));
    }

    #[test]
    fn tool_response_uses_gemma_native_inline_format() {
        let r = rag_tool_response("the Pequod set sail");
        // Inline block (no turn markers) in gemma-4's own format; the snippet is quote-delimited.
        assert_eq!(
            r,
            "<|tool_response>response:search{value:<|\"|>the Pequod set sail<|\"|>}<tool_response|>"
        );
        assert!(!r.contains("<|turn>")); // continues the model turn, doesn't open a new one
    }
}
