import json
import os
# Force PyTorch to use CPU by hiding CUDA devices
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
import glob
from pathlib import Path
from bs4 import BeautifulSoup
import pymupdf  # PyMuPDF
import fitz
from PIL import Image

try:
    from chandra.model import InferenceManager
    from chandra.input import load_pdf_images
    from chandra.model.schema import BatchInputItem
except ImportError:
    print("Please make sure you have activated the virtual environment where chandra-ocr is installed.")
    sys.exit(1)

def render_html_to_pdf_page(page, html_content: str, rect: fitz.Rect, color=(1, 0, 0)):
    soup = BeautifulSoup(html_content, "html.parser")
    
    table = soup.find("table")
    if table:
        rows = table.find_all("tr")
        if not rows:
            text = soup.get_text(separator=" ", strip=True)
            page.insert_textbox(rect, text, fontsize=6, color=color)
            return
            
        num_rows = len(rows)
        max_cols = 1
        for row in rows:
            cols = sum([int(cell.get("colspan", 1)) for cell in row.find_all(["td", "th"])])
            if cols > max_cols:
                max_cols = cols
                
        if max_cols == 0: max_cols = 1
        
        row_height = rect.height / num_rows
        col_width = rect.width / max_cols
        
        y = rect.y0
        for row in rows:
            cells = row.find_all(["td", "th"])
            x = rect.x0
            for cell in cells:
                colspan = int(cell.get("colspan", 1))
                cell_width = col_width * colspan
                
                cell_rect = fitz.Rect(x, y, x + cell_width, y + row_height)
                
                for br in cell.find_all("br"):
                    br.replace_with("\n")
                cell_text = cell.get_text(separator=" ", strip=True)
                
                if cell_text:
                    lines = cell_text.count('\n') + 1
                    fs = min(9.0, max(2.0, (cell_rect.height / lines) * 0.6))
                    rc = -1
                    while fs >= 2 and rc < 0:
                        # Align 0 (Left), as handwriting naturally starts left
                        rc = page.insert_textbox(cell_rect, cell_text, fontsize=fs, color=color, align=0)
                        if rc < 0:
                            fs *= 0.8
                x += cell_width
            y += row_height
            
    else:
        for br in soup.find_all("br"):
            br.replace_with("\n")
        for tag in soup.find_all(["p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6"]):
            tag.insert_after("\n")
            
        text = soup.get_text(separator="", strip=False)
        lines = [line.strip() for line in text.split('\n')]
        text = "\n".join([line for line in lines if line])
        import re
        text = re.sub(r'[ \t]{2,}', '  ', text)
        
        if text:
            lines = text.count('\n') + 1
            fs = min(8.0, max(2.0, (rect.height / lines) * 0.6))
            rc = -1
            while fs >= 2 and rc < 0:
                rc = page.insert_textbox(rect, text, fontsize=fs, color=color, align=0)
                if rc < 0:
                    fs *= 0.8
            if rc < 0:
                page.insert_text(rect.bottom_left, text, fontsize=2, fontname="helv", color=color)

def process_pdf(input_pdf: str, output_pdf: str, manager: InferenceManager):
    print(f"Processing {input_pdf}...")
    
    try:
        doc = fitz.open(input_pdf)
    except Exception as e:
        print(f"Failed to open PDF {input_pdf}: {e}")
        return

    # Load images for all pages
    try:
        images = load_pdf_images(input_pdf, [])
    except Exception as e:
        print(f"Failed to load images from {input_pdf}: {e}")
        doc.close()
        return
        
    if len(images) != len(doc):
        print(f"Warning: Number of images ({len(images)}) does not match number of pages ({len(doc)}).")

    for page_num in range(len(doc)):
        print(f"  Processing page {page_num + 1}/{len(doc)}...")
        page = doc[page_num]
        pdf_rect = page.rect
        pdf_width, pdf_height = pdf_rect.width, pdf_rect.height
        
        if page_num >= len(images):
            break
            
        img = images[page_num]
        img_width, img_height = img.size
        
        # Scale factors from image to PDF coords
        scale_x = pdf_width / img_width
        scale_y = pdf_height / img_height
        
        batch = [BatchInputItem(image=img, prompt_type="ocr_layout")]
        
        try:
            # Check if there's a cached json for this page
            cache_file = f"{input_pdf}_page_{page_num}.json"
            chunks = None
            if os.path.exists(cache_file):
                print(f"    Loading cached OCR data from {cache_file}...")
                with open(cache_file, "r") as f:
                    chunks = json.load(f)
            else:
                results = manager.generate(batch)
                page_result = results[0]
                chunks = page_result.chunks
                # Save cache
                with open(cache_file, "w") as f:
                    json.dump(chunks, f)
            
            for chunk in chunks:
                bbox = chunk['bbox'] # [x0, y0, x1, y1]
                content = chunk['content']
                label = chunk['label']
                
                # Skip pure images without text
                if label in ["Image", "Figure"] and not content:
                    continue
                    
                # Map bbox to PDF coordinates
                pdf_bbox = [
                    bbox[0] * scale_x,
                    bbox[1] * scale_y,
                    bbox[2] * scale_x,
                    bbox[3] * scale_y
                ]
                
                rect = fitz.Rect(pdf_bbox)
                
                try:
                    render_html_to_pdf_page(page, content, rect, color=(1, 0, 0))
                except Exception as e:
                    print(f"    Failed to render HTML at {rect}: {e}")
                    
        except Exception as e:
            import traceback
            print(f"  Error processing page {page_num + 1}:")
            traceback.print_exc()

    # Save output
    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_pdf)), exist_ok=True)
        doc.save(output_pdf)
        print(f"Saved searchable PDF to {output_pdf}")
    except Exception as e:
        print(f"Failed to save {output_pdf}: {e}")
    finally:
        doc.close()


def main():
    print("=== Chandra PDF OCR Tool ===")
    input_dir = input("Enter the input directory containing PDF files: ").strip()
    
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return
        
    output_dir = input("Enter the output directory for processed PDFs: ").strip()
    
    pdf_files = []
    # Recursively find PDFs
    for root, _, files in os.walk(input_dir):
        # Prevent searching the output directory
        if os.path.abspath(root).startswith(os.path.abspath(output_dir)):
            continue
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(root, file))
                
    if not pdf_files:
        print(f"No PDF files found in '{input_dir}'.")
        return
        
    print(f"Found {len(pdf_files)} PDF(s) to process.")
    
    # Initialize model
    print("Loading Chandra OCR model... (This may take a moment)")
    
    # Increase accuracy by rendering PDFs at a higher resolution
    from chandra.settings import settings
    settings.IMAGE_DPI = 400
    settings.MIN_PDF_IMAGE_DIM = 2048
    settings.MIN_IMAGE_DIM = 3072
    
    # Ensure PyTorch memory optimization for long runs
    os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Fallback to CPU if no GPU available (handled by PyTorch/Transformers)
    import torch
    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        print("Warning: No GPU detected, running on CPU. This will be very slow!")
        
    try:
        manager = InferenceManager(method="hf")
    except Exception as e:
        print(f"Failed to initialize InferenceManager: {e}")
        return

    for input_pdf in pdf_files:
        rel_path = os.path.relpath(input_pdf, input_dir)
        output_pdf = os.path.join(output_dir, rel_path)
        
        # Skip if output already exists?
        # if os.path.exists(output_pdf):
        #    print(f"Skipping {input_pdf}, output already exists.")
        #    continue
            
        process_pdf(input_pdf, output_pdf, manager)
        
    print("Done!")

if __name__ == "__main__":
    main()
