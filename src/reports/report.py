from pathlib import Path
from datetime import datetime, timezone
from jinja2 import Template
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

import json

def iso_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def render_md(template_path: str, context: dict) -> str:
    tpl = Path(template_path).read_text(encoding='utf-8')
    return Template(tpl).render(**context)

def save_markdown(out_path: str, md_text: str):
    p = Path(out_path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(md_text, encoding='utf-8')

def save_json(out_path: str, data: dict):
    p = Path(out_path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding='utf-8')

def try_make_pdf(md_context: dict, images: list, out_pdf: str):
    """Minimal PDF: header text + images. No md parsing to keep deps light."""
    try:
        c = canvas.Canvas(out_pdf, pagesize=A4)
        width, height = A4
        margin = 40
        y = height - margin

        # Header
        c.setFont('Helvetica-Bold', 14)
        c.drawString(margin, y, f"Deepfake Forensic Report — Case {md_context.get('case_id','')}" )
        y -= 20
        c.setFont('Helvetica', 10)
        c.drawString(margin, y, f"Created: {md_context.get('created_utc','')}  Analyst: {md_context.get('analyst','')}  Org: {md_context.get('org','')}")
        y -= 30

        # Summary
        c.setFont('Helvetica-Bold', 12)
        c.drawString(margin, y, "Summary"); y -= 16
        c.setFont('Helvetica', 10)
        lines = [
            f"Input: {md_context.get('input_path','')}",
            f"SHA-256: {md_context.get('sha256','')}",
            f"Decision: {md_context.get('decision','')} (p_fake={md_context.get('prob_fake',0):.3f})",
            f"Threshold: {md_context.get('threshold','')}",
        ]
        for ln in lines:
            c.drawString(margin, y, ln); y -= 14
        y -= 8

        # Images (heatmaps)
        for img in images:
            try:
                ir = ImageReader(img)
                iw, ih = ir.getSize()
                scale = min((width-2*margin)/iw, (height/2)/ih)
                w, h = iw*scale, ih*scale
                if y - h < margin:
                    c.showPage(); y = height - margin
                c.drawImage(ir, margin, y-h, width=w, height=h)
                y -= h + 10
            except Exception:
                continue

        c.showPage(); c.save()
        return True
    except Exception:
        return False

def generate_report(out_dir: str, template_md: str, evidence: dict):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    # Build a compact context for markdown
    ctx = {
        'case_id': evidence.get('case', {}).get('id', ''),
        'created_utc': evidence.get('created_utc',''),
        'analyst': evidence.get('case', {}).get('analyst',''),
        'org': evidence.get('case', {}).get('org',''),
        'input_path': evidence.get('input',{}).get('path',''),
        'sha256': evidence.get('input',{}).get('sha256',''),
        'duration': evidence.get('input',{}).get('video_info',{}).get('duration_sec',None),
        'decision': evidence.get('results',{}).get('decision',''),
        'prob_fake': evidence.get('results',{}).get('prob_fake',0.0),
        'threshold': evidence.get('model',{}).get('threshold',0.5),
        'tool_versions': evidence.get('tool_versions',{}),
        'heatmaps': [h.get('path') for h in evidence.get('results',{}).get('explanations',[]) if h.get('path')],
        'limitations': evidence.get('notes',{}).get('limitations',[]),
    }

    # Save evidence.json (as provided)
    save_json(str(out / 'evidence.json'), evidence)

    # Render and save Markdown
    md_text = render_md(template_md, ctx)
    save_markdown(str(out / 'report.md'), md_text)

    # Attempt to build a simple PDF
    pdf_ok = try_make_pdf(ctx, ctx['heatmaps'], str(out / 'report.pdf'))
    return {'markdown': str(out / 'report.md'), 'pdf': (str(out / 'report.pdf') if pdf_ok else None), 'json': str(out / 'evidence.json')}
