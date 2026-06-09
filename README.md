# MagicG*e*o

<sub><sup>Training-Free Text-Guided Geometric Diagram Generation</sup></sub>

MagicGeo is a training-free framework for generating geometric diagrams from textual descriptions. It formulates diagram generation as a coordinate optimization problem, integrating large language models for text translation and a formal language solver to ensure geometric correctness.

<p align="center">
  <img src="image/3.png" alt="Framework of MagicGeo">
</p>
<p align="center"><b>Figure 1.</b> Framework of MagicGeo</p>

## Performance
<p align="center">
  <img src="image/0.png" alt="example" style="max-width:100%;">
</p>
<p align="center"><b>Figure 2.</b> example</p>

![Framework of MagiceGeo](image/1.png)
![Framework of MagiceGeo](image/2.png)

## Table of Contents

- [Install](#install)
- [Usage](#usage)
- [Configuration](#configuration)
- [Datasets](#datasets)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [Maintainers](#maintainers)

## Install

Requires [uv](https://docs.astral.sh/uv/) and Python >=3.10.

```bash
uv sync
```

> **System dependency:** `pdf2image` requires [poppler](https://github.com/oschwartz10612/poppler-windows/releases/).
> - macOS: `brew install poppler`
> - Linux: `apt-get install poppler-utils`

## Configuration

Set environment variables for your API credentials:

```bash
cp .env.example .env
# Then edit .env with your API key and base URL
```

The `.env` file is automatically loaded by `python-dotenv` at runtime.

Or export them directly:

```bash
export API_KEY=your_api_key_here
export BASE_URL=https://api.deepseek.com
export MODEL_NAME=deepseek-v4-flash
```

### Using DeepSeek V4 Flash

The default configuration uses DeepSeek V4 Flash:

| Variable | Default | Description |
|---|---|---|
| `API_KEY` | (required) | Your API key |
| `BASE_URL` | `https://api.deepseek.com` | API endpoint |
| `MODEL_NAME` | `deepseek-v4-flash` | Model name |

You can also use other OpenAI-compatible providers (e.g., Qwen, OpenAI, etc.) by changing `BASE_URL` and `MODEL_NAME`.

## Usage

```bash
uv run -m geo.text_to_geometric
```

Run a single question by ID:

```bash
uv run -m geo.text_to_geometric ../json/circle.json 0
```

Arguments: `<json_path>` (default: `../json/circle.json`) `<question_id>` (omit to run all).

## Datasets

- circle: `json/circle.json`
- triangle: `json/triangle.json`
- quadrangle: `json/quadrangle.json`

## Citation

If MagicGeo has been beneficial for your research or applications, please cite it as:

```bibtex
@article{wang2025magicgeo,
  title={MagicGeo: Training-Free Text-Guided Geometric Diagram Generation},
  author={Wang, Junxiao and Zhang, Ting and Yu, Heng and Wang, Jingdong and Huang, Hua},
  journal={arXiv preprint arXiv:2502.13855},
  year={2025}
}
```

## Acknowledgments

The implementation is largely based on [DeepSeek](https://github.com/deepseek-ai/DeepSeek-V3).

## Maintainers

[@wjx](https://github.com/wjx421)
