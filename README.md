<div align="center">
  <img src="./assets/dolphin.png" width="300">
</div>

<div align="center">
  <a href="https://arxiv.org/abs/2505.14059">
    <img src="https://img.shields.io/badge/Paper-arXiv-red">
  </a>
  <a href="https://huggingface.co/ByteDance/Dolphin-1.5">
    <img src="https://img.shields.io/badge/HuggingFace-Dolphin-yellow">
  </a>
  <a href="https://github.com/bytedance/Dolphin">
    <img src="https://img.shields.io/badge/Code-Github-green">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-lightgray">
  </a>
  <br>
</div>

<br>

<div align="center">
  <img src="./assets/demo.gif" width="800">
</div>

# Dolphin: Document Image Parsing via Heterogeneous Anchor Prompting

Dolphin (**Do**cument Image **P**arsing via **H**eterogeneous Anchor Prompt**in**g) is a novel multimodal document image parsing model (**0.3B**) following an analyze-then-parse paradigm. This repository contains the demo code and pre-trained models for Dolphin.

## 📑 Overview

Document image parsing is challenging due to its complexly intertwined elements such as text paragraphs, figures, formulas, and tables. Dolphin addresses these challenges through a two-stage approach:

1. **🔍 Stage 1**: Comprehensive page-level layout analysis by generating element sequence in natural reading order
2. **🧩 Stage 2**: Efficient parallel parsing of document elements using heterogeneous anchors and task-specific prompts

<div align="center">
  <img src="./assets/framework.png" width="680">
</div>

Dolphin achieves promising performance across diverse page-level and element-level parsing tasks while ensuring superior efficiency through its lightweight architecture and parallel parsing mechanism.

<!-- ## 🚀 Demo
Try our demo on [Demo-Dolphin](https://huggingface.co/spaces/ByteDance/Dolphin). -->

## 📅 Changelog
- 🔥 **2025.10.16** Released *Dolphin-1.5* model. While maintaining the lightweight 0.3B architecture, this version achieves significant parsing improvements. (Dolphin 1.0 moved to [v1.0 branch](https://github.com/bytedance/Dolphin/tree/v1.0))
- 🔥 **2025.07.10** Released the *Fox-Page Benchmark*, a manually refined subset of the original [Fox dataset](https://github.com/ucaslcl/Fox). Download via: [Baidu Yun](https://pan.baidu.com/share/init?surl=t746ULp6iU5bUraVrPlMSw&pwd=fox1) | [Google Drive](https://drive.google.com/file/d/1yZQZqI34QCqvhB4Tmdl3X_XEvYvQyP0q/view?usp=sharing).
- 🔥 **2025.06.30** Added [TensorRT-LLM support](https://github.com/bytedance/Dolphin/blob/master/deployment/tensorrt_llm/ReadMe.md) for accelerated inference！
- 🔥 **2025.06.27** Added [vLLM support](https://github.com/bytedance/Dolphin/blob/master/deployment/vllm/ReadMe.md) for accelerated inference！
- 🔥 **2025.06.13** Added multi-page PDF document parsing capability.
- 🔥 **2025.05.21** Our demo is released at [link](http://115.190.42.15:8888/dolphin/). Check it out!
- 🔥 **2025.05.20** The pretrained model and inference code of Dolphin are released.
- 🔥 **2025.05.16** Our paper has been accepted by ACL 2025. Paper link: [arXiv](https://arxiv.org/abs/2505.14059).

## 📈 Performance

<table style="width:90%; border-collapse: collapse; text-align: center;">
    <caption>Comprehensive evaluation of document parsing on Fox-Page and Dolphin-Page</caption>
    <thead>
        <tr>
            <th style="text-align: center !important;">Model</th>
            <th style="text-align: center !important;">Fox-Page-en<sup>Edit</sup>&#x2193;</th>
            <th style="text-align: center !important;">Fox-Page-zh<sup>Edit</sup>&#x2193;</th>
            <th style="text-align: center !important;">Dolphin-Page-<sup>Edit</sup>&#x2193;</th>
            <th style="text-align: center !important;">Evg<sup>Edit</sup>&#x2193;</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Dolphin</td>
            <td>0.0114</td>
            <td>0.0131</td>
            <td>0.1028</td>
            <td>0.0424</td>
        </tr>
        <tr>
            <td>Dolphin-1.5</td>
            <td><strong>0.0074</strong></td>
            <td><strong>0.0077</strong></td>
            <td><strong>0.0743</strong></td>
            <td><strong>0.0298</strong></td>
        </tr>
    </tbody>
</table>

<table style="width:90%; border-collapse: collapse; text-align: center;">
    <caption>Comprehensive evaluation of document parsing on OmniDocBench (v1.5)</caption>
    <thead>
        <tr>
            <th style="text-align: center !important;">Model</th>
            <th style="text-align: center !important;">Overall&#x2191;</th>
            <th style="text-align: center !important;">Text<sup>Edit</sup>&#x2193;</th>
            <th style="text-align: center !important;">Formula<sup>CDM</sup>&#x2191;</th>
            <th style="text-align: center !important;">Table<sup>TEDS</sup>&#x2191;</th>
            <th style="text-align: center !important;">Table<sup>TEDS-S</sup>&#x2191;</th>
            <th style="text-align: center !important;">Read Order<sup>Edit</sup>&#x2193;</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Dolphin</td>
            <td>74.67</td>
            <td>0.125</td>
            <td>67.85</td>
            <td>68.70</td>
            <td>77.77</td>
            <td>0.124</td>
        </tr>
        <tr>
            <td>Dolphin-1.5</td>
            <td><strong>83.21</strong></td>
            <td><strong>0.092</strong></td>
            <td><strong>80.78</strong></td>
            <td><strong>78.06</strong></td>
            <td><strong>84.10</strong></td>
            <td><strong>0.080</strong></td>
        </tr>
    </tbody>
</table>

## 🛠️ Installation

### Prerequisites

**Python 3.11 or higher** is required. Install using UV (recommended):
```bash
# Install UV first
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Then install Python 3.11+ with UV
uv python install 3.11
```

**Alternative Python Installation:**
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install python3.11 python3.11-venv

# macOS (using Homebrew)
brew install python@3.11

# Windows - Download from python.org
```

### Quick Installation (Recommended)

1. Clone the repository:
```bash
   git clone https://github.com/ByteDance/Dolphin.git
   cd Dolphin
```

2. Create and activate a virtual environment with UV:
```bash
   uv venv
   # Activate the virtual environment:
   # Linux/macOS:
   source .venv/bin/activate
   # Windows:
   .venv\Scripts\activate
```

3. Run the automated installation script:
```bash
   python install.py
```
   
   **Installation Options:**
   - `--cpu`: Install CPU-only version of PyTorch (no CUDA support)
   - `--skip-model`: Skip automatic model download
```bash
   # Example: CPU-only installation
   python install.py --cpu
   
   # Example: Install without downloading model
   python install.py --skip-model
```

   **GPU Requirements:**
   - GPU with CUDA or ROCm 12.4 support for GPU acceleration
   - Automatic fallback to CPU if GPU is not detected
   - macOS users will get MPS (Metal Performance Shaders) support automatically

### Manual Installation (Alternative)

If you prefer manual control over the installation process:

1. **Install PyTorch** (choose based on your system):
```bash
   # For CUDA 12.4 (Linux/Windows with GPU)
   uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   
   # For CPU-only
   uv pip install torch torchvision torchaudio
   
   # For macOS (MPS support)
   uv pip install torch torchvision torchaudio
```

2. **Install dependencies**:
```bash
   uv pip install -e .
```

3. **Download the model**:
```bash
   huggingface-cli download ByteDance/Dolphin-1.5 --local-dir ./hf_model --local-dir-use-symlinks False
```

### Legacy Installation

For backwards compatibility, you can still use the traditional pip-based installation:
```bash
# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Download model manually (see Manual Installation step 3 above)
```

### 🎨 Quick Demo with Gradio

Launch the interactive Gradio web interface:
```bash
# Make sure your virtual environment is activated first!
python dolphin_parser_app.py
```

**Important:** Do NOT use `uv run` for running scripts after installation, as it creates a fresh isolated environment and will reinstall packages (losing your CUDA-enabled PyTorch). Always activate the virtual environment first:
```bash
# Activate venv (if not already activated)
# Linux/macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# Then run the script
python dolphin_parser_app.py
```

This will start a local web server (typically at `http://127.0.0.1:7860`) where you can:
- Upload document images or PDFs
- View parsed results exported as markdown

### 🔧 Troubleshooting

**CUDA not detected: (assuming nvidia, change for amd)**
- Verify GPU driver installation: `nvidia-smi`
- Check CUDA version compatibility (requires CUDA 12.4)
- Use `--cpu` flag if GPU is not available

**Model download fails:**
- Check internet connection and Hugging Face access
- Manually download from [Hugging Face](https://huggingface.co/ByteDance/Dolphin-1.5)
- Ensure sufficient disk space (~2GB required)

**Out of memory errors:**
- Reduce `--max_batch_size` in page-level parsing
- Use CPU mode with `--cpu` flag
- Close other applications to free up memory

**Gradio port already in use:**
- Gradio will automatically try alternative ports
- Or manually specify: `uv run dolphin_parser_app.py --server-port 7861`

**UV command not found after installation:**
- Restart your terminal/shell
- Add to PATH manually: `export PATH="$HOME/.cargo/bin:$PATH"`


## ⚡ Inference

Dolphin provides two inference frameworks with support for two parsing granularities:
- **Page-level Parsing**: Parse the entire document page into a structured JSON and Markdown format
- **Element-level Parsing**: Parse individual document elements (text, table, formula)


### 📄 Page-level Parsing

```bash
# Process a single document image
python demo_page.py --model_path ./hf_model --save_dir ./results \
    --input_path ./demo/page_imgs/page_1.png 

# Process a single document pdf
python demo_page.py --model_path ./hf_model --save_dir ./results \
    --input_path ./demo/page_imgs/page_6.pdf 

# Process all documents in a directory
python demo_page.py --model_path ./hf_model --save_dir ./results \
    --input_path ./demo/page_imgs 

# Process with custom batch size for parallel element decoding
python demo_page.py --model_path ./hf_model --save_dir ./results \
    --input_path ./demo/page_imgs \
    --max_batch_size 8
```

### 🧩 Element-level Parsing

````bash
# Process element images (specify element_type: table, formula, text, or code)
python demo_element.py --model_path ./hf_model --save_dir ./results \
    --input_path  \
    --element_type [table|formula|text|code]
````

### 🎨 Layout Parsing
````bash
# Process a single document image
python demo_layout.py --model_path ./hf_model --save_dir ./results \
    --input_path ./demo/page_imgs/page_1.png \
    
# Process a single PDF document
python demo_layout.py --model_path ./hf_model --save_dir ./results \
    --input_path ./demo/page_imgs/page_6.pdf \

# Process all documents in a directory
python demo_layout.py --model_path ./hf_model --save_dir ./results \
    --input_path ./demo/page_imgs 
````


## 🌟 Key Features

- 🔄 Two-stage analyze-then-parse approach based on a single VLM
- 📊 Promising performance on document parsing tasks
- 🔍 Natural reading order element sequence generation
- 🧩 Heterogeneous anchor prompting for different document elements
- ⏱️ Efficient parallel parsing mechanism
- 🤗 Support for Hugging Face Transformers for easier integration


## 📮 Notice
**Call for Bad Cases:** If you have encountered any cases where the model performs poorly, we would greatly appreciate it if you could share them in the issue. We are continuously working to optimize and improve the model.

## 💖 Acknowledgement

We would like to acknowledge the following open-source projects that provided inspiration and reference for this work:
- [OmniDocBench](https://github.com/opendatalab/OmniDocBench)
- [Donut](https://github.com/clovaai/donut/)
- [Nougat](https://github.com/facebookresearch/nougat)
- [GOT](https://github.com/Ucas-HaoranWei/GOT-OCR2.0)
- [MinerU](https://github.com/opendatalab/MinerU/tree/master)
- [Swin](https://github.com/microsoft/Swin-Transformer)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)

## 📝 Citation

If you find this code useful for your research, please use the following BibTeX entry.

```bibtex
@article{feng2025dolphin,
  title={Dolphin: Document Image Parsing via Heterogeneous Anchor Prompting},
  author={Feng, Hao and Wei, Shu and Fei, Xiang and Shi, Wei and Han, Yingdong and Liao, Lei and Lu, Jinghui and Wu, Binghong and Liu, Qi and Lin, Chunhui and others},
  journal={arXiv preprint arXiv:2505.14059},
  year={2025}
}
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=bytedance/Dolphin&type=Date)](https://www.star-history.com/#bytedance/Dolphin&Date)
