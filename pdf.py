import json
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER

def format_tree(node, indent=0):
    """Recursively walks the JSON tree to generate formatted ReportLab paragraphs."""
    elements = []
    styles = getSampleStyleSheet()
    
    # Layout-aware font indentation parameters
    dir_style = ParagraphStyle(
        'DirStyle',
        parent=styles['Heading3'],
        leftIndent=indent * 15,
        textColor=colors.HexColor('#1A365D'),
        spaceBefore=4,
        spaceAfter=2
    )
    
    file_style = ParagraphStyle(
        'FileStyle',
        parent=styles['Normal'],
        leftIndent=indent * 15,
        textColor=colors.HexColor('#2D3748'),
        fontName='Helvetica-Bold',
        spaceBefore=2,
        spaceAfter=2
    )
    
    item_style = ParagraphStyle(
        'ItemStyle',
        parent=styles['Normal'],
        leftIndent=(indent + 1) * 15,
        textColor=colors.HexColor('#4A5568'),
        fontName='Courier',
        fontSize=9,
        spaceBefore=1,
        spaceAfter=1
    )

    if node.get('node_type') == 'directory':
        elements.append(Paragraph(f"<b>{node['name']}/</b>", dir_style))
        for child in node.get('children', []):
            elements.extend(format_tree(child, indent + 1))
    elif node.get('node_type') == 'file':
        elements.append(Paragraph(f"{node['name']}", file_style))
        for item in node.get('items', []):
            item_type = item.get('type', 'item').upper()
            item_name = item.get('name', '')
            # Clean character strings ensuring zero translation layout bugs
            elements.append(Paragraph(f"&bull; <b>{item_type}</b>: {item_name}", item_style))
            
    return elements

def main():
    json_file = "codebase_structure.json"
    output_pdf = "codebase_structure.pdf"
    
    if not os.path.exists(json_file):
        print(f"Error: Target data map '{json_file}' not found.")
        return

    print(f"Reading {json_file}...")
    with open(json_file, 'r') as f:
        json_data = json.load(f)

    print("Compiling PDF tree layouts...")
    doc = SimpleDocTemplate(
        output_pdf,
        pagesize=letter,
        rightMargin=54, leftMargin=54, topMargin=54, bottomMargin=54,
        title="Codebase Structure Report"
    )
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        alignment=TA_CENTER,
        textColor=colors.HexColor('#2B6CB0'),
        spaceAfter=15
    )
    
    story = [
        Paragraph("Codebase Architecture &amp; AST Map", title_style),
        Spacer(1, 10)
    ]
    
    story.extend(format_tree(json_data))
    doc.build(story)
    print(f"Successfully generated PDF at: {os.path.abspath(output_pdf)}")

if __name__ == "__main__":
    main()
