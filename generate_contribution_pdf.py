from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Preformatted
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from datetime import datetime

# Create PDF
pdf_path = "APS_Sensor_Individual_Contribution.pdf"
doc = SimpleDocTemplate(pdf_path, pagesize=letter,
                        rightMargin=0.75*inch, leftMargin=0.75*inch,
                        topMargin=0.75*inch, bottomMargin=0.75*inch)

elements = []

# Styles
styles = getSampleStyleSheet()
title_style = ParagraphStyle(
    'CustomTitle',
    parent=styles['Heading1'],
    fontSize=22,
    textColor=colors.HexColor('#1F4E78'),
    spaceAfter=6,
    alignment=TA_CENTER,
    fontName='Helvetica-Bold'
)

heading_style = ParagraphStyle(
    'CustomHeading',
    parent=styles['Heading2'],
    fontSize=14,
    textColor=colors.white,
    backColor=colors.HexColor('#2E5C8A'),
    spaceAfter=12,
    spaceBefore=12,
    leftIndent=10,
    fontName='Helvetica-Bold'
)

subheading_style = ParagraphStyle(
    'CustomSubHeading',
    parent=styles['Heading3'],
    fontSize=11,
    textColor=colors.HexColor('#2E5C8A'),
    spaceAfter=8,
    spaceBefore=8,
    fontName='Helvetica-Bold'
)

normal_style = ParagraphStyle(
    'CustomNormal',
    parent=styles['Normal'],
    fontSize=10,
    alignment=TA_JUSTIFY,
    spaceAfter=6
)

bullet_style = ParagraphStyle(
    'CustomBullet',
    parent=styles['Normal'],
    fontSize=10,
    alignment=TA_LEFT,
    spaceAfter=4,
    leftIndent=20
)

# Title
elements.append(Paragraph("3. INDIVIDUAL CONTRIBUTION & SYSTEM OVERVIEW", title_style))
elements.append(Spacer(1, 0.1*inch))

intro_text = """This section describes the individual contribution to the APS Sensor Fault Prediction project along with 
a comprehensive overview of the complete machine learning system. This section explains the system architecture, 
component integration, challenges faced during implementation, and the solutions deployed to ensure reliable model performance."""

elements.append(Paragraph(intro_text, normal_style))
elements.append(Spacer(1, 0.15*inch))

# Section 3.1
elements.append(Paragraph("3.1 INDIVIDUAL CONTRIBUTION", heading_style))
elements.append(Spacer(1, 0.08*inch))

elements.append(Paragraph("<b>Role: ML Pipeline Architecture & Backend Implementation</b>", subheading_style))
role_text = """In this project, the primary responsibility was to design and implement the complete machine learning 
pipeline infrastructure. The pipeline layer forms the foundation of the system, as accurate predictions depend on proper 
data processing, model training, and seamless component integration."""

elements.append(Paragraph(role_text, normal_style))
elements.append(Spacer(1, 0.1*inch))

# 3.1.1
elements.append(Paragraph("3.1.1 System Architecture Design", subheading_style))
arch_text = """The first step was planning the ML pipeline architecture to create a scalable, modular, and reliable system 
that can handle data ingestion, validation, transformation, model training, evaluation, and deployment."""

elements.append(Paragraph(arch_text, normal_style))

elements.append(Paragraph("<b>The architecture included:</b>", bullet_style))

arch_items = [
    "Data Ingestion Module: Fetching sensor data from MongoDB collections",
    "Data Validation Module: Ensuring data quality and schema compliance",
    "Data Transformation Module: Feature engineering and preprocessing",
    "Model Training Module: XGBoost classifier implementation with hyperparameter tuning",
    "Model Evaluation Module: Performance metrics and model selection",
    "Model Pusher Module: Deployment to cloud storage (S3)",
    "Cloud Integration: AWS S3 bucket syncing and MongoDB database connectivity"
]

for item in arch_items:
    elements.append(Paragraph(f"• {item}", bullet_style))

elements.append(Spacer(1, 0.1*inch))

# 3.1.2
elements.append(Paragraph("3.1.2 Data Pipeline Integration", subheading_style))
integration_text = """Each component was designed to work seamlessly with the next stage in the pipeline. Care was taken 
to ensure that data flows correctly through each stage without loss of information or integrity."""

elements.append(Paragraph(integration_text, normal_style))
elements.append(Paragraph("<b>The integration involved:</b>", bullet_style))

integration_items = [
    "Data Ingestion: Connecting to MongoDB and exporting sensor data as DataFrames",
    "Schema Validation: Validating data against YAML configuration schemas",
    "Feature Engineering: Creating meaningful features from raw sensor readings",
    "Train-Test Splitting: Ensuring proper data stratification",
    "Model Integration: Seamlessly passing processed data to the training module"
]

for item in integration_items:
    elements.append(Paragraph(f"• {item}", bullet_style))

elements.append(Spacer(1, 0.1*inch))

# 3.1.3
elements.append(Paragraph("3.1.3 Configuration Management & Environment Setup", subheading_style))
config_text = """Proper configuration management was critical for system reliability and flexibility. The implementation 
included centralized configuration, secure environment variables, and comprehensive logging infrastructure."""

elements.append(Paragraph(config_text, normal_style))
elements.append(Paragraph("<b>Implementation included:</b>", bullet_style))

config_items = [
    "YAML-based Configuration: Separate config files for database, S3, and training pipeline",
    "Environment Variables: Secure storage of credentials and API keys",
    "Modular Constants: Centralized application constants for easy maintenance",
    "Logging Infrastructure: Comprehensive logging across all pipeline stages",
    "Exception Handling: Custom SensorException for graceful error management"
]

for item in config_items:
    elements.append(Paragraph(f"• {item}", bullet_style))

elements.append(PageBreak())

# 3.1.4
elements.append(Paragraph("3.1.4 Database & Cloud Storage Integration", subheading_style))
cloud_text = """The system was designed to handle data persistence and cloud deployment with multi-layer storage 
approach ensuring data safety and model reproducibility."""

elements.append(Paragraph(cloud_text, normal_style))
elements.append(Paragraph("<b>Integration components:</b>", bullet_style))

cloud_items = [
    "MongoDB Connection: Secure connections to MongoDB Atlas clusters",
    "Data Export: Converting MongoDB collections to Pandas DataFrames",
    "S3 Integration: Syncing trained models and artifacts to AWS S3 buckets",
    "Artifact Management: Organizing training artifacts in timestamped directories",
    "SSL/TLS Security: Using certified connections for secure data transmission"
]

for item in cloud_items:
    elements.append(Paragraph(f"• {item}", bullet_style))

elements.append(Spacer(1, 0.1*inch))

# 3.1.5
elements.append(Paragraph("3.1.5 Model Training & Evaluation Pipeline", subheading_style))
model_text = """Implementing the machine learning components required careful orchestration to ensure optimal model 
performance and validation."""

elements.append(Paragraph(model_text, normal_style))
elements.append(Paragraph("<b>Components implemented:</b>", bullet_style))

model_items = [
    "Data Loading: Efficiently loading transformed data from feature stores",
    "Feature Scaling: Normalizing features for optimal model performance",
    "XGBoost Implementation: Configuring and training the gradient boosting classifier",
    "Hyperparameter Tuning: Fine-tuning model parameters for better accuracy",
    "Performance Metrics: Computing precision, recall, F1-score, and AUC-ROC",
    "Model Comparison: Comparing trained model with baseline models"
]

for item in model_items:
    elements.append(Paragraph(f"• {item}", bullet_style))

elements.append(Spacer(1, 0.1*inch))

# 3.1.6
elements.append(Paragraph("3.1.6 Challenges Faced", subheading_style))
elements.append(Paragraph("<b>Technical challenges encountered:</b>", bullet_style))

challenges = [
    "Data Quality Issues: Handling missing values, outliers, and data inconsistencies",
    "Schema Mismatch: Ensuring consistency between data and defined schemas",
    "Memory Constraints: Processing large datasets efficiently",
    "Pipeline Failures: Handling failures without affecting entire pipeline",
    "Model Overfitting: Balancing model complexity with generalization",
    "Cloud Connectivity: Ensuring reliable connections to cloud services"
]

for challenge in challenges:
    elements.append(Paragraph(f"• {challenge}", bullet_style))

elements.append(Spacer(1, 0.1*inch))

# 3.1.7
elements.append(Paragraph("3.1.7 Solutions Implemented", subheading_style))
elements.append(Paragraph("<b>Solutions deployed:</b>", bullet_style))

solutions = [
    "Data Preprocessing: Robust handling of missing values and outliers",
    "Validation Framework: Comprehensive data validation at each pipeline stage",
    "Batch Processing: Efficient batch processing for large datasets",
    "Error Recovery: Checkpoints and retry mechanisms for pipeline resilience",
    "Model Regularization: L1/L2 regularization and early stopping",
    "Redundant Connections: Connection pooling and automatic reconnection logic"
]

for solution in solutions:
    elements.append(Paragraph(f"• {solution}", bullet_style))

elements.append(Spacer(1, 0.1*inch))

# 3.1.8
elements.append(Paragraph("3.1.8 Outcome of Contribution", subheading_style))
outcome_text = """Through this work, a complete and production-ready ML pipeline was successfully developed. 
All components were properly integrated and tested, delivering accurate fault predictions."""

elements.append(Paragraph(outcome_text, normal_style))
elements.append(Paragraph("<b>The pipeline implementation ensured:</b>", bullet_style))

outcomes = [
    "Automated Data Processing: End-to-end data flow without manual intervention",
    "Model Reproducibility: Consistent results across different runs and environments",
    "Scalability: Ability to handle larger datasets and multiple model versions",
    "Production Readiness: Cloud deployment capability with proper monitoring"
]

for outcome in outcomes:
    elements.append(Paragraph(f"• {outcome}", bullet_style))

elements.append(PageBreak())

# Section 3.2
elements.append(Paragraph("3.2 SYSTEM OVERVIEW", heading_style))
elements.append(Spacer(1, 0.08*inch))

overview_text = """The developed system is an end-to-end machine learning solution designed to predict Air Pressure System 
(APS) sensor failures in heavy-duty vehicles. It consists of multiple interconnected components working together in a 
structured data pipeline."""

elements.append(Paragraph(overview_text, normal_style))
elements.append(Spacer(1, 0.1*inch))

# System Architecture
elements.append(Paragraph("<b>System Architecture</b>", subheading_style))
arch_overview = """The system uses sensor data collected from APS systems, processes it through multiple validation and 
transformation stages, trains an XGBoost classifier model, and deploys it to the cloud for predictions."""

elements.append(Paragraph(arch_overview, normal_style))

# Flow diagram
flow_text = """<b>Complete System Flow:</b><br/>Raw Data (MongoDB) → Data Ingestion → Data Validation → Data Transformation 
→ Model Training → Model Evaluation → Model Push (S3) → Deployment"""

elements.append(Paragraph(flow_text, bullet_style))
elements.append(Spacer(1, 0.1*inch))

# Component Description
elements.append(Paragraph("<b>Component Description</b>", subheading_style))
elements.append(Spacer(1, 0.05*inch))

components = [
    ("Data Ingestion", "Extracts sensor readings from MongoDB database and exports them as feature stores."),
    ("Data Validation", "Validates incoming data against predefined schemas, ensuring data quality."),
    ("Data Transformation", "Performs feature engineering, scaling, and preprocessing for model training."),
    ("Model Training", "Trains an XGBoost classifier with optimized hyperparameters."),
    ("Model Evaluation", "Evaluates model performance using precision, recall, F1-score, and AUC-ROC."),
    ("Model Pusher", "Saves the trained model and artifacts to AWS S3 for deployment.")
]

for comp_name, comp_desc in components:
    elements.append(Paragraph(f"<b>{comp_name}:</b> {comp_desc}", normal_style))

elements.append(Spacer(1, 0.1*inch))

# Testing Results
elements.append(Paragraph("<b>System Testing & Results</b>", subheading_style))
elements.append(Paragraph("""The system was tested using actual APS sensor data. The testing confirmed:""", normal_style))

results = [
    "Data Processing: Successful ingestion and transformation of 36,188+ sensor records",
    "Model Performance: Achieving high accuracy in fault classification",
    "System Stability: Reliable execution across multiple pipeline runs",
    "Cloud Integration: Seamless synchronization with AWS S3 and MongoDB"
]

for result in results:
    elements.append(Paragraph(f"• {result}", bullet_style))

elements.append(Spacer(1, 0.15*inch))

# Conclusion
conclusion_text = """The working architecture demonstrates a professional-grade ML systems design, suitable for 
production deployment in critical infrastructure monitoring applications. The system combines robust data processing, 
advanced machine learning techniques, and cloud integration to deliver reliable APS sensor fault predictions."""

elements.append(Paragraph(conclusion_text, normal_style))
elements.append(Spacer(1, 0.2*inch))

# Footer
footer_text = "APS Sensor Fault Prediction Project — Individual Contribution & System Overview"
elements.append(Paragraph(footer_text, ParagraphStyle('footer', parent=styles['Normal'], fontSize=9, 
                                                       alignment=TA_CENTER, textColor=colors.grey)))

# Build PDF
doc.build(elements)
print(f"✓ Individual Contribution & System Overview PDF generated successfully!")
print(f"✓ File saved as: {pdf_path}")
