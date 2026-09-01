# Place int8 E5 artifacts here when inmotion-social is not a sibling checkout.

Required files (from `intfloat/multilingual-e5-small`):

- `model_qint8_avx512_vnni.onnx`  (~118 MB) — or set `E5_MODEL_PATH`
- `sentencepiece.bpe.model`
- `config.json`
- `tokenizer_config.json`

Override directory with `E5_MODEL_DIR`.
