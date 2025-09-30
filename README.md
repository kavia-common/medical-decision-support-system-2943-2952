# medical-decision-support-system-2943-2952

Django-based Medical Decision Support backend with a multi-agent architecture.

Features:
- PatientAgent chatbot collects structured medical data with PHI redaction and red-flag detection.
- ClinicalAgent performs RAG retrieval using a local vector store and generates provisional recommendations.
- OneDrive integration placeholder with automatic local fallback for session notes and file uploads.
- REST API endpoints:
  - GET /api/health/
  - POST /api/chat/
  - POST /api/recommend/
  - POST /api/upload_report/
  - GET /api/get_recommendation/?session_id=...
- CLI fallback: `python manage.py cli_demo`
- Seed RAG demo docs: `python manage.py seed_rag`

API Docs:
- Swagger UI: /docs/
- ReDoc: /redoc/