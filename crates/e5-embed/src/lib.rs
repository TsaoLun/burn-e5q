//! E5-small int8 embedding pipeline on burn, mirroring inmotion-social's
//! `rec::text_embed` (ort) behaviour:
//!
//! - sentencepiece tokenizer with HF id remapping (bos→0, eos→2, unk→3, else +1)
//! - `"passage: "` / `"query: "` prefixes, empty text → zero vector
//! - per-forward token budget packing (rows × padded_len ≤ budget, longest first)
//! - mean pooling with attention mask, then L2 normalize to 384 dims

pub mod model {
    include!(concat!(env!("OUT_DIR"), "/model/model_qint8_avx512_vnni.rs"));
}

use std::path::{Path, PathBuf};

use anyhow::{Context, bail};
use burn::prelude::*;
use sentencepiece_rs::SentencePieceProcessor;

pub const TEXT_EMB_DIM: usize = 384;
pub const E5_PASSAGE_PREFIX: &str = "passage: ";
pub const E5_QUERY_PREFIX: &str = "query: ";

const DEFAULT_MAX_LENGTH: usize = 512;
const DEFAULT_BATCH_SIZE: usize = 256;
const DEFAULT_MAX_BATCH_TOKENS: usize = 4096;

/// `tokenizer.json` inserts `<pad>` at id 1 and moves `<unk>` 0→3;
/// normal SentencePiece pieces are shifted by +1 in HF/ONNX ids.
const HF_SPACE_TOKEN_ID: i64 = 6;
const SENTENCEPIECE_ADDED_TOKENS: [(&str, i64); 5] = [
    ("</s>", 2),
    ("<mask>", 250001),
    ("<pad>", 1),
    ("<unk>", 3),
    ("<s>", 0),
];

pub struct E5Embedder {
    model: model::Model,
    tokenizer: E5Tokenizer,
    device: Device,
    max_batch_tokens: usize,
}

struct E5Tokenizer {
    processor: SentencePieceProcessor,
    max_length: usize,
    pad_id: i64,
    bos_id: i64,
    eos_id: i64,
    unk_id: i64,
}

impl E5Tokenizer {
    fn load(model_dir: &Path, max_length: usize) -> anyhow::Result<Self> {
        let config: serde_json::Value = serde_json::from_slice(
            &std::fs::read(model_dir.join("config.json")).context("read config.json")?,
        )
        .context("parse config.json")?;
        let pad_id = config["pad_token_id"].as_u64().unwrap_or(0) as i64;

        let model_path = model_dir.join("sentencepiece.bpe.model");
        let processor = SentencePieceProcessor::open(&model_path)
            .map_err(|e| anyhow::anyhow!("load {}: {e}", model_path.display()))?;
        let bos_id = processor
            .bos_id()
            .context("sentencepiece model missing bos id")? as i64;
        let eos_id = processor
            .eos_id()
            .context("sentencepiece model missing eos id")? as i64;
        let unk_id = processor.unk_id() as i64;

        Ok(Self {
            processor,
            max_length,
            pad_id,
            bos_id,
            eos_id,
            unk_id,
        })
    }

    fn hf_id(&self, sentencepiece_id: usize) -> i64 {
        let id = sentencepiece_id as i64;
        if id == self.bos_id {
            0
        } else if id == self.eos_id {
            2
        } else if id == self.unk_id {
            3
        } else {
            id + 1
        }
    }

    fn encode_plain(&self, text: &str) -> anyhow::Result<Vec<i64>> {
        let mut ids = self
            .processor
            .encode_to_ids(text)
            .map_err(|e| anyhow::anyhow!("sentencepiece encode: {e}"))?
            .into_iter()
            .map(|id| self.hf_id(id))
            .collect::<Vec<_>>();

        // SentencePiece drops a trailing space; HF keeps it as id 6.
        if text.chars().last().is_some_and(char::is_whitespace)
            && ids.last().copied() != Some(HF_SPACE_TOKEN_ID)
        {
            ids.push(HF_SPACE_TOKEN_ID);
        }
        Ok(ids)
    }

    fn encode_text(&self, text: &str) -> anyhow::Result<Vec<i64>> {
        let mut out = Vec::new();
        let mut offset = 0;

        while offset < text.len() {
            let next = SENTENCEPIECE_ADDED_TOKENS
                .iter()
                .filter_map(|(token, id)| {
                    text[offset..]
                        .find(token)
                        .map(|pos| (offset + pos, offset + pos + token.len(), *id))
                })
                .min_by_key(|(start, end, _)| (*start, std::cmp::Reverse(*end)));

            let Some((start, end, id)) = next else {
                out.extend(self.encode_plain(&text[offset..])?);
                break;
            };

            if start > offset {
                out.extend(self.encode_plain(&text[offset..start])?);
            }
            out.push(id);
            offset = end;
        }

        Ok(out)
    }

    /// Per-text token ids with bos/eos, truncated, **no padding**.
    fn encode_rows(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<i64>>> {
        let max_piece_count = self.max_length.saturating_sub(2);
        let mut rows = Vec::with_capacity(texts.len());
        for text in texts {
            let ids = self.encode_text(text)?;
            let mut row = Vec::with_capacity(ids.len().min(max_piece_count) + 2);
            row.push(self.hf_id(self.bos_id as usize));
            row.extend(ids.into_iter().take(max_piece_count));
            row.push(self.hf_id(self.eos_id as usize));
            rows.push(row);
        }
        Ok(rows)
    }
}

/// Partition row indices into encoder batches, longest rows first, under the
/// per-forward token budget (`rows × padded_len ≤ max_batch_tokens`).
fn pack_batches(row_lens: &[usize], max_batch_tokens: usize) -> Vec<Vec<usize>> {
    let mut order: Vec<usize> = (0..row_lens.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(row_lens[i]));

    let mut batches: Vec<Vec<usize>> = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    let mut padded_len = 0usize;
    for idx in order {
        let fits = !current.is_empty()
            && current.len() < DEFAULT_BATCH_SIZE
            && (current.len() + 1) * padded_len <= max_batch_tokens;
        if current.is_empty() {
            padded_len = row_lens[idx].max(1);
            current.push(idx);
        } else if fits {
            current.push(idx);
        } else {
            batches.push(std::mem::take(&mut current));
            padded_len = row_lens[idx].max(1);
            current.push(idx);
        }
    }
    if !current.is_empty() {
        batches.push(current);
    }
    batches
}

struct BatchEncoding {
    input_ids: Vec<i64>,
    attention_mask: Vec<i64>,
    token_type_ids: Vec<i64>,
    batch_size: usize,
    encoding_length: usize,
}

fn pad_rows(rows: &[Vec<i64>], indices: &[usize], pad_id: i64) -> BatchEncoding {
    let batch_size = indices.len();
    let encoding_length = indices
        .iter()
        .map(|&i| rows[i].len())
        .max()
        .unwrap_or(0)
        .max(1);

    let mut input_ids = vec![pad_id; batch_size * encoding_length];
    let mut attention_mask = vec![0i64; batch_size * encoding_length];
    let token_type_ids = vec![0i64; batch_size * encoding_length];

    for (row_idx, &i) in indices.iter().enumerate() {
        let row = &rows[i];
        let start = row_idx * encoding_length;
        input_ids[start..start + row.len()].copy_from_slice(row);
        attention_mask[start..start + row.len()].fill(1);
    }

    BatchEncoding {
        input_ids,
        attention_mask,
        token_type_ids,
        batch_size,
        encoding_length,
    }
}

/// Mean pooling with attention mask (E5 / sentence-transformers convention),
/// then L2 normalize per row.
fn mean_pool_l2(
    last_hidden_state: Tensor<3>,
    attention_mask: Tensor<2, Int>,
) -> Tensor<2> {
    let mask = attention_mask.float().unsqueeze_dim::<3>(2);
    let masked = last_hidden_state * mask.clone();
    let sum = masked.sum_dim(1);
    // Rows are never empty (bos+eos guarantee sum ≥ 2), clamp only guards 0.
    let sum_mask = mask.sum_dim(1).clamp_min(1e-9);
    let pooled = (sum / sum_mask).squeeze_dim::<2>(1);

    let norm = pooled
        .clone()
        .powf_scalar(2.0)
        .sum_dim(1)
        .sqrt()
        .clamp_min(1e-12);
    pooled / norm
}

impl E5Embedder {
    pub fn load(model_dir: &Path, device: &Device) -> anyhow::Result<Self> {
        let weights = concat!(env!("OUT_DIR"), "/model/model_qint8_avx512_vnni.bpk");
        let model = model::Model::from_file(weights, device);
        let tokenizer = E5Tokenizer::load(model_dir, DEFAULT_MAX_LENGTH)?;
        Ok(Self {
            model,
            tokenizer,
            device: device.clone(),
            max_batch_tokens: DEFAULT_MAX_BATCH_TOKENS,
        })
    }

    /// Override the per-forward token budget (rows × padded_len ≤ budget).
    pub fn with_max_batch_tokens(mut self, budget: usize) -> Self {
        self.max_batch_tokens = budget;
        self
    }

    /// Embed passages (post bodies). Empty/whitespace → zero vector.
    pub fn embed_passages(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        self.embed_prefixed(E5_PASSAGE_PREFIX, texts)
    }

    /// Embed queries (asymmetric retrieval). Empty/whitespace → zero vector.
    pub fn embed_queries(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        self.embed_prefixed(E5_QUERY_PREFIX, texts)
    }

    /// Raw sentencepiece ids for a prefixed text (used to verify tokenizer
    /// parity against the HF tokenizer in the reference data).
    pub fn encode_prefixed(&self, prefix: &str, text: &str) -> anyhow::Result<Vec<i64>> {
        let full = format!("{prefix}{}", text.trim());
        let rows = self.tokenizer.encode_rows(&[full.as_str()])?;
        Ok(rows.into_iter().next().unwrap_or_default())
    }

    fn embed_prefixed(&self, prefix: &str, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let mut out = vec![vec![0.0f32; TEXT_EMB_DIM]; texts.len()];
        let mut owned: Vec<String> = Vec::new();
        let mut indices: Vec<usize> = Vec::new();
        for (i, text) in texts.iter().enumerate() {
            let trimmed = text.trim();
            if trimmed.is_empty() {
                continue;
            }
            owned.push(format!("{prefix}{trimmed}"));
            indices.push(i);
        }
        if owned.is_empty() {
            return Ok(out);
        }
        let refs: Vec<&str> = owned.iter().map(String::as_str).collect();
        let raw = self.forward_all(&refs)?;
        for (idx, v) in indices.into_iter().zip(raw) {
            out[idx] = v;
        }
        Ok(out)
    }

    /// Encode all texts in budget-packed sub-batches; output order = input order.
    fn forward_all(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        let rows = self.tokenizer.encode_rows(texts)?;
        let row_lens: Vec<usize> = rows.iter().map(Vec::len).collect();
        let pad_id = self.tokenizer.pad_id;

        let mut out: Vec<Vec<f32>> = vec![Vec::new(); texts.len()];
        for batch in pack_batches(&row_lens, self.max_batch_tokens) {
            let encoded = pad_rows(&rows, &batch, pad_id);
            let vectors = self.forward_batch(encoded);
            if vectors.len() != batch.len() {
                bail!(
                    "encoder returned {} vectors for a batch of {}",
                    vectors.len(),
                    batch.len()
                );
            }
            for (&i, v) in batch.iter().zip(vectors) {
                out[i] = v;
            }
        }
        Ok(out)
    }

    /// Raw model forward without pooling (for diagnostics).
    pub fn forward_raw(
        &self,
        input_ids: Tensor<2, Int>,
        attention_mask: Tensor<2, Int>,
        token_type_ids: Tensor<2, Int>,
    ) -> Tensor<3> {
        self.model.forward(input_ids, attention_mask, token_type_ids)
    }

    fn forward_batch(&self, encoded: BatchEncoding) -> Vec<Vec<f32>> {
        let shape = [encoded.batch_size, encoded.encoding_length];
        let input_ids =
            Tensor::<2, Int>::from_data(TensorData::new(encoded.input_ids, shape), &self.device);
        let attention_mask = Tensor::<2, Int>::from_data(
            TensorData::new(encoded.attention_mask, shape),
            &self.device,
        );
        let token_type_ids = Tensor::<2, Int>::from_data(
            TensorData::new(encoded.token_type_ids, shape),
            &self.device,
        );

        let last_hidden = self
            .model
            .forward(input_ids, attention_mask.clone(), token_type_ids);
        let pooled = mean_pool_l2(last_hidden, attention_mask);

        pooled
            .into_data()
            .try_to_vec::<f32>()
            .expect("pooled embedding dtype")
            .chunks(TEXT_EMB_DIM)
            .map(|c| c.to_vec())
            .collect()
    }
}

pub fn default_model_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("E5_MODEL_DIR") {
        return PathBuf::from(dir);
    }
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let sibling = manifest.join("../../../inmotion-social/data/models/multilingual-e5-small");
    if sibling.join("sentencepiece.bpe.model").is_file() {
        return sibling;
    }
    manifest.join("models")
}

pub fn current_rss_mb() -> f64 {
    memory_stats::memory_stats()
        .map(|u| u.physical_mem as f64 / (1024.0 * 1024.0))
        .unwrap_or(0.0)
}
