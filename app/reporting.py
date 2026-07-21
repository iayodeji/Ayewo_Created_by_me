import io
from typing import Iterable

# Fits filename text into the PDF table column width on letter-sized pages.
MAX_FILENAME_LENGTH = 38


def build_batch_pdf(batch_id: str, rows: Iterable[dict]) -> bytes:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    y = height - 50
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(40, y, f"Ayewo Batch Report: {batch_id}")
    y -= 30

    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(40, y, "Filename")
    pdf.drawString(280, y, "Result")
    pdf.drawString(390, y, "Confidence")
    y -= 16

    pdf.setFont("Helvetica", 10)
    for row in rows:
        if y < 40:
            pdf.showPage()
            y = height - 40
            pdf.setFont("Helvetica", 10)
        pdf.drawString(40, y, str(row["filename"])[:MAX_FILENAME_LENGTH])
        pdf.drawString(280, y, str(row["result"]))
        pdf.drawString(390, y, f'{float(row["confidence"]):.2f}%')
        y -= 14

    pdf.save()
    return buffer.getvalue()
