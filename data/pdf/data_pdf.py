import os
import logging
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
from faker import Faker

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

fake = Faker()


def sanitize_filename(name):
    """Create a filesystem-friendly filename from a string."""
    return "".join(c for c in name if c.isalnum() or c in (' ', '_')).rstrip().replace(" ", "_")


def generate_medical_report(patient_info, diagnosis, treatment_plan):
    """Generate a formatted medical report PDF and save it in the data/pdf/ directory."""

    output_dir = os.path.join("data", "pdf")
    os.makedirs(output_dir, exist_ok=True)

    styles = getSampleStyleSheet()
    title_style = styles["Heading1"]
    title_style.fontSize = 20
    title_style.alignment = 1
    title_style.textColor = colors.darkblue

    section_style = styles["Heading2"]
    section_style.fontSize = 14
    section_style.textColor = colors.darkred

    body_style = styles["BodyText"]

    elements = []

    # Add logo if available
    logo_path = "hospital_logo.png"
    if os.path.exists(logo_path):
        elements.append(Image(logo_path, width=100, height=50))

    elements = [Paragraph("Medical Report", title_style), Spacer(1, 10)]

    patient_table_data = [
        ["Patient Name", patient_info['name']],
        ["Age", patient_info['age']],
        ["Gender", patient_info['gender']],
        ["Marital Status", patient_info.get('marital_status', 'N/A')],
        ["Medical History", patient_info.get('medical_history', 'N/A')],
        ["Family History", patient_info.get('family_history', 'N/A')]
    ]

    patient_table = Table(patient_table_data, style=[
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ])
    elements.append(patient_table)

    elements += [
        Spacer(1, 10),
        Paragraph("Diagnosis:", section_style),
        Paragraph(diagnosis, styles["BodyText"]),
        Spacer(1, 10),
        Paragraph("Treatment Plan:", section_style),
        Paragraph(treatment_plan, styles["BodyText"]),
        Spacer(1, 20),
        Paragraph("Doctor's Signature: ____________________", styles["Normal"]),
        Spacer(1, 5),
        Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d')}", styles["Normal"]),
    ]

    report_filename = f"{sanitize_filename(patient_info['name'])}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    output_dir = os.path.join(os.path.dirname(__file__))
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, report_filename)

    doc = SimpleDocTemplate(file_path, pagesize=A4)
    doc.build(elements)

    logging.info(f"Report successfully generated and saved at {file_path}")


if __name__ == "__main__":
    patient_data = {
        "name": fake.name(),
        "age": fake.random_int(min=18, max=90),
        "gender": fake.random_element(elements=('Male', 'Female')),
        "marital_status": fake.random_element(elements=('Single', 'Married', 'Divorced', 'Widowed')),
        "medical_history": fake.sentence(),
        "family_history": fake.sentence()
    }

    diagnosis_text = fake.sentence(nb_words=3)
    treatment_plan_text = "\n".join([f"- {fake.sentence()}" for _ in range(2)])

    generate_medical_report(patient_data, diagnosis_text, treatment_plan_text)