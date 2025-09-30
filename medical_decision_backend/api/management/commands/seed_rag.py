from django.core.management.base import BaseCommand
from api.rag import LocalVectorStore


DOCS = [
    {"id": "cardiology_acs", "text": "Patients with acute chest pain should be evaluated for acute coronary syndrome with ECG and troponin testing.", "meta": {"source": "guideline:cardiology"}},
    {"id": "pulmonary_sob", "text": "Shortness of breath requires assessment of oxygen saturation and consideration of imaging if pneumonia or heart failure suspected.", "meta": {"source": "guideline:pulmonology"}},
    {"id": "neuro_headache", "text": "Red flags for headache include sudden severe onset, neurologic deficits, fever, and altered mental status.", "meta": {"source": "guideline:neurology"}},
    {"id": "allergy_anaphylaxis", "text": "Anaphylaxis presents with airway compromise; administer epinephrine and seek emergency care immediately.", "meta": {"source": "guideline:allergy"}},
    {"id": "general_history", "text": "A thorough history includes onset, duration, severity, associated symptoms, medical history, medications, and allergies.", "meta": {"source": "guideline:general"}},
]


class Command(BaseCommand):
    help = "Seed the local RAG index with demo guideline passages."

    def handle(self, *args, **options):
        vs = LocalVectorStore()
        vs.add_documents(DOCS)
        self.stdout.write(self.style.SUCCESS(f"Added {len(DOCS)} documents to local RAG index."))
