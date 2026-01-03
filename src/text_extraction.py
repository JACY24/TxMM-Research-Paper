import fitz  # PyMuPDF
import os

def get_sorted_text(page):
    """
    Extracts text blocks and sorts them based on a two-column layout heuristic.
    Logic:
    1. Identify Header/Footer regions (top/bottom 5-10% of page).
    2. Split the middle region into Left and Right columns based on the page center.
    3. Sort blocks: Header -> Left Col -> Right Col -> Footer.
    """
    blocks = page.get_text("blocks")
    # block format: (x0, y0, x1, y1, "text", block_no, block_type)
    
    width = page.rect.width
    height = page.rect.height
    
    # Define zones
    header_threshold = height * 0.08  # Top 8% is header
    footer_threshold = height * 0.92  # Bottom 8% is footer
    center_x = width / 2

    # Buckets for text blocks
    headers = []
    footers = []
    left_col = []
    right_col = []

    for b in blocks:
        x0, y0, x1, y1, text, block_no, block_type = b
        
        # skip images (block_type 1)
        if block_type != 0:
            continue

        # Sort into buckets
        if y1 < header_threshold:
            headers.append(b)
        elif y0 > footer_threshold:
            footers.append(b)
        else:
            # Body text: Decided left/right based on starting X coordinate
            if x0 < center_x:
                left_col.append(b)
            else:
                right_col.append(b)

    # Sort each bucket internally by vertical position (y0)
    # Note: We use a small tolerance for X so lines aligned roughly same height are ordered correctly
    headers.sort(key=lambda b: b[1])
    left_col.sort(key=lambda b: b[1])
    right_col.sort(key=lambda b: b[1])
    footers.sort(key=lambda b: b[1])

    # Combine all text
    sorted_blocks = headers + left_col + right_col + footers

    final_paragraphs = []
    current_para = ""

    for b in sorted_blocks:
        # Get text and remove existing newlines within the block itself
        text = b[4].replace('\n', ' ').strip()
        
        if not text:
            continue
            
        # If we have no current paragraph started, start one
        if not current_para:
            current_para = text
            continue
            
        # Check: Should we merge this block with the previous one?
        
        # 1. Does the previous block look like an unfinished sentence? 
        #    (Doesn't end with . ! ? or :)
        prev_ends_abruptly = not current_para.endswith(('.', '!', '?', ':'))
        
        # 2. Does the current block look like a continuation?
        #    (Starts with lowercase)
        curr_starts_lower = text[0].islower()
   
        # 3. Check for hyphenation (e.g. "ver- kiezing" -> "verkiezing")
        if current_para.endswith('-'):
            current_para = current_para[:-1] + text # Merge without space
            
        elif prev_ends_abruptly or curr_starts_lower:
            # Merge with a space
            current_para += " " + text
            
        else:
            # Previous sentence ended and new one looks like a start. 
            # Save the old one and start a new one.
            final_paragraphs.append(current_para)
            current_para = text

    # Append the very last paragraph remaining in the buffer
    if current_para:
        final_paragraphs.append(current_para)
    
    # Return text with double newlines between paragraphs
    return "\n\n".join(final_paragraphs)

def process_manifestos(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = [f for f in os.listdir(input_folder) if f.lower().endswith('.pdf')]
    print(f"Found {len(files)} PDFs. Starting smart extraction...\n")

    for filename in files:
        pdf_path = os.path.join(input_folder, filename)
        txt_path = os.path.join(output_folder, filename.replace('.pdf', '.txt'))
        
        print(f"Processing: {filename}")
        
        try:
            doc = fitz.open(pdf_path)
            all_text = []

            for i, page in enumerate(doc):
                # Extract using the smart column logic
                page_text = get_sorted_text(page)
                
                # Add page marker
                all_text.append(f"--- Page {i+1} ---")
                all_text.append(page_text)
                all_text.append("\n" + "="*20 + "\n")
            
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(all_text))
                
            print(f"Saved: {txt_path}\n")
            
        except Exception as e:
            print(f"Error reading {filename}: {e}")