from django.core.management.base import BaseCommand
from api.utils import generate_session_id
from api.storage import HybridStorage
from api.rag import LocalVectorStore
from api.agents import PatientAgent, ClinicalAgent


class Command(BaseCommand):
    help = "Run a CLI demo of the medical decision support multi-agent system."

    def handle(self, *args, **options):
        storage = HybridStorage()
        vs = LocalVectorStore()
        patient = PatientAgent(storage)
        clinical = ClinicalAgent(storage, vs)

        session_id = generate_session_id()
        self.stdout.write(self.style.SUCCESS(f"New session: {session_id}"))
        self.stdout.write("Type messages to answer the PatientAgent questions. Type '/done' when finished, '/rec' for recommendation.")

        while True:
            try:
                msg = input("> ").strip()
            except EOFError:
                break
            if not msg:
                continue
            if msg == "/done":
                break
            if msg == "/rec":
                rec = clinical.recommend(session_id, extra_notes="")
                self.stdout.write(self.style.SUCCESS("Recommendation:"))
                self.stdout.write(f"{rec}")
                continue
            result = patient.handle_message(session_id, msg)
            self.stdout.write(f"Agent: {result.get('message')} (complete={result.get('complete')})")
        self.stdout.write(self.style.SUCCESS("CLI demo ended."))
