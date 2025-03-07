import random
import os
import re
from datetime import datetime, timedelta
from faker import Faker
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

fake = Faker()

def generate_patient_data():
    name = fake.name()
    age = random.randint(18, 90)
    gender = random.choice(["Male", "Female"])
    marital_status = random.choice(["Single", "Married", "Divorced"])
    medical_history = random.choice(["Hypertension", "Diabetes", "Asthma", "None"])
    family_history = random.choice(["Family history of diabetes", "No family history", "Family history of heart disease"])
    
    diagnosis = random.choice([
        "Hypertension", "Type 2 Diabetes", "Asthma", "Chronic Fatigue Syndrome", "Hypothyroidism"
    ])
    
    treatment_plan = random.choice([
        "Daily medication for 3 months.",
        "Dietary changes and exercise.",
        "Regular check-ups every 6 months.",
        "Surgical intervention recommended."
    ])
    
    appointment_date = datetime.today() + timedelta(days=random.randint(1, 30))
    follow_up_date = appointment_date + timedelta(days=60)
    
    inquiries = [
        "How can I manage my blood sugar levels effectively?",
        "Are there any lifestyle changes I can make?",
        "What are the side effects of my medication?",
        "Should I be concerned about weight gain from medication?"
    ]
    
    patient_info = {
        "name": name,
        "age": age,
        "gender": gender,
        "marital_status": marital_status,
        "medical_history": medical_history,
        "family_history": family_history,
        "diagnosis": diagnosis,
        "treatment_plan": treatment_plan,
        "appointment_date": appointment_date.strftime("%Y-%m-%d"),
        "follow_up_date": follow_up_date.strftime("%Y-%m-%d"),
        "inquiries": random.sample(inquiries, 2)  
    }
    
    return patient_info

def generate_disease_specific_text(disease):
    """Generate a long text related to the disease."""
    disease_texts = {
        "Hypertension": "Hypertension, also known as high blood pressure, is a condition where the force of the blood against the artery walls is consistently too high. Hypertension can lead to serious health complications such as heart disease, stroke, and kidney failure. It is often referred to as the 'silent killer' because it may not show obvious symptoms until it's too late. Lifestyle changes, including a healthy diet, regular physical activity, and reducing stress, are crucial for managing hypertension. Medications such as ACE inhibitors, beta-blockers, and diuretics are commonly prescribed to control blood pressure levels.",
        
        "Type 2 Diabetes": "Type 2 diabetes is a chronic condition that affects the way the body metabolizes sugar (glucose). It occurs when the body becomes resistant to insulin or doesn't produce enough insulin, leading to elevated blood glucose levels. Over time, high blood sugar levels can cause damage to the heart, kidneys, eyes, and nerves. Managing diabetes involves a combination of lifestyle changes, including eating a balanced diet, exercising regularly, and monitoring blood glucose levels. Medications such as metformin and insulin may also be used to help control blood sugar levels.",
        
        "Asthma": "Asthma is a chronic respiratory disease characterized by inflammation and narrowing of the airways, which makes it difficult to breathe. Symptoms of asthma include wheezing, coughing, shortness of breath, and chest tightness. Asthma can be triggered by various factors such as allergens, respiratory infections, physical activity, and environmental pollutants. The goal of asthma management is to control symptoms and prevent attacks. This is typically done through the use of inhalers that contain bronchodilators or corticosteroids, as well as avoiding triggers.",
        
        "Chronic Fatigue Syndrome": "Chronic Fatigue Syndrome (CFS), also known as myalgic encephalomyelitis, is a condition characterized by persistent and unexplained fatigue that doesn't improve with rest. In addition to fatigue, symptoms can include muscle pain, difficulty sleeping, memory problems, and headaches. The exact cause of CFS is unknown, but it is thought to involve a combination of viral infections, immune system problems, and genetic factors. Treatment focuses on managing symptoms and improving quality of life, including lifestyle modifications and cognitive behavioral therapy.",
        
        "Hypothyroidism": "Hypothyroidism is a condition in which the thyroid gland doesn't produce enough thyroid hormone, leading to a slow metabolism and a variety of symptoms. Symptoms can include fatigue, weight gain, constipation, dry skin, and depression. The most common cause of hypothyroidism is an autoimmune disorder called Hashimoto's thyroiditis, where the immune system attacks the thyroid gland. Treatment typically involves hormone replacement therapy with synthetic thyroid hormone, which helps to normalize hormone levels and alleviate symptoms."
    }

    # Return the disease-specific text or a default message if the disease is not found
    return disease_texts.get(disease, "This disease description is not available.")

def generate_medical_report(patient_info):
    output_dir = "data/pdf/"
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
    
    sanitized_name = re.sub(r'\W+', '_', patient_info['name'])  # Clean up the patient name for filename
    file_path = os.path.join(output_dir, f"medical_report_{sanitized_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")

    doc = SimpleDocTemplate(file_path, pagesize=A4)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Title"],
        fontSize=20,
        spaceAfter=15,
        textColor=colors.darkblue,
        alignment=1
    )

    section_style = ParagraphStyle(
        "SectionStyle",
        parent=styles["Heading2"],
        fontSize=14,
        textColor=colors.darkred,
        spaceAfter=10,
        underline=True
    )

    body_style = ParagraphStyle(
        "BodyStyle",
        parent=styles["BodyText"],
        fontSize=12,
        leading=18,
        spaceAfter=12
    )

    elements = []

    logo_path = "hospital_logo.png"  
    if os.path.exists(logo_path):
        elements.append(Image(logo_path, width=100, height=50))

    elements.append(Paragraph("Medical Report: Clinical Case Review", title_style))
    elements.append(Spacer(1, 5))
    
    report_date = datetime.now().strftime("%Y-%m-%d")
    elements.append(Paragraph(f"Date: {report_date}", styles["Normal"]))
    elements.append(Spacer(1, 10))

    patient_data = [
        ["Patient Name", patient_info['name']],
        ["Age", str(patient_info['age'])],
        ["Gender", patient_info['gender']],
        ["Marital Status", patient_info['marital_status']],
        ["Medical History", patient_info['medical_history']],
        ["Family History", patient_info['family_history']]
    ]
    
    table = Table(patient_data, colWidths=[150, 250])
    table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                               ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                               ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                               ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                               ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                               ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                               ('GRID', (0, 0), (-1, -1), 1, colors.black)]))

    elements.append(table)
    elements.append(Spacer(1, 15))

    elements.append(Paragraph("Diagnosis:", section_style))
    elements.append(Paragraph(patient_info['diagnosis'], body_style))
    elements.append(Spacer(1, 12))

    # Generate disease-specific text to reach 10,000 words
    disease_text = generate_disease_specific_text(patient_info['diagnosis'])
    long_text = disease_text * 20  # Multiply the text to get around 10,000 words

    elements.append(Paragraph("Disease Details:", section_style))
    elements.append(Paragraph(long_text, body_style))  # Add the long disease-specific text
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Treatment Plan:", section_style))
    elements.append(Paragraph(patient_info['treatment_plan'], body_style))
    elements.append(Spacer(1, 12))

    elements.append(Spacer(1, 20))
    elements.append(Paragraph("Doctor's Signature: ____________________", styles["Normal"]))
    elements.append(Spacer(1, 5))
    elements.append(Paragraph(f"Date: {report_date}", styles["Normal"]))

    elements.append(Spacer(1, 20))
    elements.append(Paragraph("Official Clinic Stamp", styles["Normal"]))

    doc.build(elements)
    print(f"The formatted medical report has been generated and saved as {file_path}")

# Example: Generate a report for a patient
patient_info = generate_patient_data()
generate_medical_report(patient_info)
