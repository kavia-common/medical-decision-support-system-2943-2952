from typing import Dict, Any, List, Optional

from .utils import redact_phi, detect_red_flags, now_iso
from .storage import HybridStorage
from .rag import LocalVectorStore


PATIENT_QUESTIONS = [
    {"key": "chief_complaint", "question": "What is your main concern today?"},
    {"key": "onset", "question": "When did it start, and has it changed over time?"},
    {"key": "severity", "question": "On a scale of 1-10, how severe is it?"},
    {"key": "associated_symptoms", "question": "Any other symptoms you've noticed?"},
    {"key": "medical_history", "question": "Do you have any significant medical history?"},
    {"key": "medications", "question": "What medications are you currently taking?"},
    {"key": "allergies", "question": "Any allergies to medications or foods?"},
]


def _next_question(collected: Dict[str, Any]) -> Optional[Dict[str, str]]:
    for q in PATIENT_QUESTIONS:
        if q["key"] not in collected or not collected[q["key"]]:
            return q
    return None


class PatientAgent:
    """
    Patient-facing agent that collects structured data through a chat-like interaction.
    Applies PHI redaction and red-flag detection. Persists conversation turns.
    """

    def __init__(self, storage: HybridStorage):
        self.storage = storage

    # PUBLIC_INTERFACE
    def handle_message(self, session_id: str, message: str) -> Dict[str, Any]:
        """
        Process a user message, update structured fields, and return next prompt.
        """
        # Redact PHI
        redacted_text, redactions = redact_phi(message)
        red_flags = detect_red_flags(redacted_text)

        # Retrieve current note
        note = self.storage.get_latest_note(session_id) or {"session_id": session_id, "turns": [], "structured": {}}
        note.setdefault("structured", {})
        note.setdefault("turns", [])
        note["turns"].append({"from": "user", "text": redacted_text, "timestamp": now_iso(), "redactions": redactions})

        # Simple heuristic mapping: if last question asked, capture answer
        last_turns = [t for t in note["turns"] if t.get("from") == "agent" and t.get("type") == "question"]
        expected_key = last_turns[-1]["key"] if last_turns else None
        if expected_key:
            note["structured"][expected_key] = redacted_text

        next_q = _next_question(note["structured"])
        if next_q:
            agent_msg = {
                "from": "agent",
                "type": "question",
                "key": next_q["key"],
                "text": next_q["question"],
                "timestamp": now_iso(),
            }
            note["turns"].append(agent_msg)
            self.storage.save_session_note(session_id, note)
            return {
                "session_id": session_id,
                "message": agent_msg["text"],
                "next_key": next_q["key"],
                "red_flags": red_flags,
                "redactions": redactions,
                "structured": note["structured"],
                "complete": False,
            }
        else:
            completion_msg = {
                "from": "agent",
                "type": "completion",
                "text": "Thank you. I have collected your basic information.",
                "timestamp": now_iso(),
            }
            note["turns"].append(completion_msg)
            self.storage.save_session_note(session_id, note)
            return {
                "session_id": session_id,
                "message": completion_msg["text"],
                "red_flags": red_flags,
                "redactions": redactions,
                "structured": note["structured"],
                "complete": True,
            }


class ClinicalAgent:
    """
    Clinical agent that uses RAG to retrieve relevant passages and generate provisional recommendations.
    """

    SAFETY_DISCLAIMER = (
        "Safety note: This output is for informational purposes only and does not replace professional medical advice. "
        "If you experience red-flag symptoms, seek emergency care immediately."
    )

    def __init__(self, storage: HybridStorage, vector_store: LocalVectorStore):
        self.storage = storage
        self.vs = vector_store

    def _build_context(self, session_id: str, extra_notes: str = "") -> str:
        note = self.storage.get_latest_note(session_id) or {}
        structured = note.get("structured", {})
        files = self.storage.list_files(session_id).get("files", [])
        file_names = ", ".join([f["filename"] for f in files]) if files else "None"
        ctx_lines = [
            f"Session: {session_id}",
            f"Chief complaint: {structured.get('chief_complaint', 'n/a')}",
            f"Onset: {structured.get('onset', 'n/a')}",
            f"Severity: {structured.get('severity', 'n/a')}",
            f"Associated symptoms: {structured.get('associated_symptoms', 'n/a')}",
            f"History: {structured.get('medical_history', 'n/a')}",
            f"Medications: {structured.get('medications', 'n/a')}",
            f"Allergies: {structured.get('allergies', 'n/a')}",
            f"Attached reports: {file_names}",
        ]
        if extra_notes:
            ctx_lines.append(f"Additional notes: {extra_notes}")
        return "\n".join(ctx_lines)

    def _retrieve_guidelines(self, query: str, k: int) -> List[Dict[str, Any]]:
        return self.vs.similarity_search(query, top_k=k)

    def _synthesize_recommendation(self, context: str, passages: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Heuristic recommendation generation.
        recs = []
        ctx_low = context.lower()
        if "chest pain" in ctx_low:
            recs.append("Consider ECG and cardiac enzymes; evaluate for acute coronary syndrome.")
        if "shortness of breath" in ctx_low or "difficulty breathing" in ctx_low:
            recs.append("Assess oxygen saturation; consider chest imaging if indicated.")
        if "fever" in ctx_low:
            recs.append("Check temperature and evaluate for signs of infection; consider CBC.")
        if "headache" in ctx_low:
            recs.append("Assess for red-flag features (sudden severe onset, neurologic deficits).")
        if "allerg" in ctx_low:
            recs.append("Review allergy details and potential triggers; consider antihistamines if appropriate.")
        if not recs:
            recs.append("Gather more history and consider basic vitals, labs, and symptom-targeted exam.")

        guidelines = [{"score": round(p["score"], 4), "excerpt": p["text"], "source": p["meta"].get("source", "local")} for p in passages]
        result = {
            "summary": "Preliminary clinical considerations based on provided information.",
            "recommendations": recs,
            "guideline_support": guidelines,
            "safety": self.SAFETY_DISCLAIMER,
        }
        return result

    # PUBLIC_INTERFACE
    def recommend(self, session_id: str, extra_notes: str = "", top_k: int = 3) -> Dict[str, Any]:
        """
        Generate a structured provisional recommendation for a session.
        """
        context = self._build_context(session_id, extra_notes)
        passages = self._retrieve_guidelines(context, k=max(1, top_k))
        recommendation = self._synthesize_recommendation(context, passages)

        # Persist as a note in the session
        self.storage.save_session_note(
            session_id,
            {
                "session_id": session_id,
                "type": "recommendation",
                "context": context,
                "result": recommendation,
            },
        )
        return recommendation
