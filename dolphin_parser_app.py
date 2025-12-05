import gradio as gr
import os
import sys
import shutil
import torch
import json
from PIL import Image
import fitz  # PyMuPDF for PDF preview

# --- Add the current directory to Python path ---
# This is necessary to import modules from the repo.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --- Import Dolphin's core processing functions ---
try:
    from demo_page import DOLPHIN, process_document
    from utils.utils import setup_output_dirs
except ImportError as e:
    print(f"Error: Could not import necessary modules: {e}")
    print("Please ensure 'dolphin_parser_app.py' is in the main 'Dolphin' directory.")
    sys.exit(1)

# --- Configuration ---
MODEL_PATH = "./hf_model"
TEMP_RESULTS_DIR = "./temp_gradio_results"

# --- Global Model Loading ---
# Instantiate the DOLPHIN model once when the script starts.
print("Loading Dolphin model... (This may take a few minutes on the first run)")

# Configure GPU usage
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    # Set CUDA memory management to reduce overhead
    torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of GPU memory
    torch.backends.cudnn.benchmark = True  # Optimize for fixed-size inputs
else:
    print("⚠️ No GPU found, using CPU (this will be slower)")

try:
    model = DOLPHIN(MODEL_PATH)
    print("✅ Dolphin model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

def results_to_markdown(results):
    """Converts the JSON results from Dolphin into a formatted Markdown string."""
    markdown_output = []
    
    # Check if it's a multi-page PDF result
    if isinstance(results, list) and results and 'page_number' in results[0]:
        # It's a list of pages
        for page in results:
            markdown_output.append(f"## Page {page['page_number']}\n")
            markdown_output.append(format_elements(page['elements']))
    else:
        # It's a single image result
        markdown_output.append(format_elements(results))
        
    return "\n".join(markdown_output)

def format_elements(elements):
    """Formats a list of document elements into Markdown."""
    md_parts = []
    for element in elements:
        label = element.get('label', 'text')
        text = element.get('text', '')
        
        if not text:
            continue
            
        if label == 'text':
            md_parts.append(f"{text}\n")
        elif label == 'equ':
            md_parts.append(f"$$\n{text}\n$$\n")
        elif label == 'code':
            md_parts.append(f"```\n{text}\n```\n")
        elif label == 'tab':
            md_parts.append(f"{text}\n") # Assuming text is already a Markdown table
        elif label == 'fig':
            md_parts.append(f"{text}\n") # Assuming text is already a Markdown image link
        else:
            md_parts.append(f"{text}\n") # Fallback for unknown labels
            
    return "\n".join(md_parts)

def get_document_preview(input_path):
    """Generate a preview image of the document."""
    try:
        file_ext = os.path.splitext(input_path)[1].lower()
        
        if file_ext == '.pdf':
            # Get first page of PDF
            doc = fitz.open(input_path)
            page = doc[0]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            doc.close()
            return img
        else:
            # For images, just open directly
            return Image.open(input_path)
    except Exception as e:
        print(f"Error generating preview: {e}")
        return None

def parse_document_and_display(input_file, progress=gr.Progress()):
    """
    Takes an uploaded file, processes it with Dolphin, and returns the results.
    """
    if model is None:
        return None, "## Error: Dolphin model could not be loaded. Please check the console output.", None, gr.Button(visible=False), gr.Button(visible=False)

    if input_file is None:
        return None, "## Please upload a document file.", None, gr.Button(visible=False), gr.Button(visible=False)

    input_path = input_file.name
    
    # Generate document preview
    progress(0.1, desc="Generating document preview...")
    preview_image = get_document_preview(input_path)
    
    # Clean up and setup temporary directory
    progress(0.2, desc="Setting up output directory...")
    if os.path.exists(TEMP_RESULTS_DIR):
        shutil.rmtree(TEMP_RESULTS_DIR)
    os.makedirs(TEMP_RESULTS_DIR, exist_ok=True)
    # This creates the 'figures' subdirectory needed by the processing script
    setup_output_dirs(TEMP_RESULTS_DIR) 
        
    print(f"Processing {input_path} with Dolphin...")
    
    try:
        # Call the correct processing function
        progress(0.3, desc="Processing document with Dolphin...")
        _, recognition_results = process_document(
            document_path=input_path,
            model=model,
            save_dir=TEMP_RESULTS_DIR,
            max_batch_size=8 # You can adjust this value
        )
        
        progress(0.8, desc="Converting to Markdown...")
        # Convert the structured results to Markdown
        markdown_content = results_to_markdown(recognition_results)
        
        # Save outputs for download
        progress(0.9, desc="Preparing downloads...")
        markdown_file = os.path.join(TEMP_RESULTS_DIR, "output.md")
        json_file = os.path.join(TEMP_RESULTS_DIR, "output.json")
        
        with open(markdown_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(recognition_results, f, ensure_ascii=False, indent=2)
        
        progress(1.0, desc="Complete!")
        print("✅ Processing complete.")
        
        return (
            preview_image,
            markdown_content,
            recognition_results,
            gr.Button(value="📥 Download Markdown", visible=True, interactive=True),
            gr.Button(value="📥 Download JSON", visible=True, interactive=True)
        )

    except Exception as e:
        print(f"An error occurred during processing: {e}")
        return (
            preview_image,
            f"## An Error Occurred\n\n```\n{e}\n```",
            None,
            gr.Button(visible=False),
            gr.Button(visible=False)
        )

def download_markdown():
    """Return the path to the markdown file for download."""
    markdown_file = os.path.join(TEMP_RESULTS_DIR, "output.md")
    if os.path.exists(markdown_file):
        return markdown_file
    return None

def download_json():
    """Return the path to the JSON file for download."""
    json_file = os.path.join(TEMP_RESULTS_DIR, "output.json")
    if os.path.exists(json_file):
        return json_file
    return None

# --- Gradio Interface ---
with gr.Blocks(title="Dolphin Document Parser", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🐬 Dolphin Document Parser
        
        Upload a PDF or image to parse its structure and content into Markdown using ByteDance's [Dolphin model](https://github.com/bytedance/Dolphin).
        """
    )
    
    # Store results in state
    results_state = gr.State()
    
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="Upload Document (PDF or Image)", 
                file_types=[".pdf", ".png", ".jpg", ".jpeg", ".tiff"],
                height=150
            )
            process_button = gr.Button("🚀 Parse Document", variant="primary", size="lg")
            
            gr.Markdown("### Document Preview")
            preview_image = gr.Image(
                label="Document Preview",
                type="pil",
                height=400,
                interactive=False
            )
            
        with gr.Column(scale=2):
            output_markdown = gr.Markdown(
                label="Parsed Markdown Output",
                value="Parsed content will appear here..."
            )
            
            with gr.Row():
                download_md_btn = gr.Button(
                    "📥 Download Markdown",
                    visible=False,
                    variant="secondary"
                )
                download_json_btn = gr.Button(
                    "📥 Download JSON",
                    visible=False,
                    variant="secondary"
                )
            
            # Hidden download components
            markdown_download = gr.File(label="Markdown File", visible=False)
            json_download = gr.File(label="JSON File", visible=False)

    # Set up the button click event with progress
    process_button.click(
        fn=parse_document_and_display,
        inputs=file_input,
        outputs=[preview_image, output_markdown, results_state, download_md_btn, download_json_btn]
    )
    
    # Download handlers
    download_md_btn.click(
        fn=download_markdown,
        outputs=markdown_download
    ).then(
        lambda x: gr.File(value=x, visible=True),
        inputs=markdown_download,
        outputs=markdown_download
    )
    
    download_json_btn.click(
        fn=download_json,
        outputs=json_download
    ).then(
        lambda x: gr.File(value=x, visible=True),
        inputs=json_download,
        outputs=json_download
    )

if __name__ == "__main__":
    # Check if model exists before launching
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at '{MODEL_PATH}'")
        print("Please follow the setup instructions to download the model.")
    else:
        # Launch the Gradio app
        demo.launch()