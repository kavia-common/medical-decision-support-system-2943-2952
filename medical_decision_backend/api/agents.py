from typing import Dict, Any, List, Optional

from .utils import (
    redact_phi,
    detect_red_flags,
    now_iso,
    extract_fields_from_text,
    build_acknowledgment,
    get_suggestions_for_key,
    get_hints_for_key,
    build_safety_banner,
    transcript_tail,
)
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

def _last_agent_question(turns: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    qs = [t for t in turns if t.get("from") == "agent" and t.get("type") == "question"]
    return qs[-1] if qs else None

class PatientAgent:
    """
    Patient-facing agent that collects structured data through a chat-like interaction.
    Applies PHI redaction and red-flag detection. Persists conversation turns.

    Conversational behavior:
    - Acknowledges user input referencing previous question.
    - Extracts multiple fields from a single message where possible.
    - Adds dynamic safety banner if red-flag terms are detected.
    - Provides suggestions and hints for the next question.
    - Returns a richer response payload while keeping legacy fields for backward compatibility.
    """

    def __init__(self, storage: HybridStorage):
        self.storage = storage

    # PUBLIC_INTERFACE
    def handle_message(self, session_id: str, message: str) -> Dict[str, Any]:
        """
        Process a user message, update structured fields, and return next prompt.

        Returns both a rich chat payload and legacy keys for compatibility.
        """
        # Redact PHI early
        redacted_text, redactions = redact_phi(message)
        red_flags = detect_red_flags(redacted_text)

        # Restore latest note state
        note = self.storage.get_latest_note(session_id) or {
            "session_id": session_id,
            "turns": [],
            "structured": {},
        }
        note.setdefault("structured", {})
        note.setdefault("turns", [])

        # Persist user turn
        user_turn = {
            "from": "user",
            "type": "message",
            "text": redacted_text,
            "timestamp": now_iso(),
            "redactions": redactions,
            "red_flags": red_flags,
        }
        note["turns"].append(user_turn)

        # Determine expected key from last asked agent question
        last_q = _last_agent_question(note["turns"])
        expected_key = last_q.get("key") if last_q else None

        # Multi-field extraction from single user message
        extracted = extract_fields_from_text(redacted_text, expected_key=expected_key)
        if extracted:
            note["structured"].update(extracted)

        # Acknowledge prior answer before asking next
        ack_text = build_acknowledgment(expected_key, redacted_text)

        # Decide next question (or completion)
        next_q = _next_question(note["structured"])

        # Optional safety banner (display before or alongside next question)
        safety_payload = build_safety_banner(red_flags)

        display_delay_ms = 550  # micro typing simulation hint for UI

        if next_q:
            # Compose conversational question with acknowledgment lead-in
            q_text = f"{ack_text} {next_q['question']}"
            suggestions = get_suggestions_for_key(next_q["key"])
            hints = get_hints_for_key(next_q["key"])

            agent_turn = {
                "from": "agent",
                "role": "assistant",
                "type": "question",
                "text": q_text,
                "key": next_q["key"],
                "timestamp": now_iso(),
                "suggestions": suggestions or None,
                "hints": hints,
            }
            # Save the exact question turn for context and key-binding
            note["turns"].append(agent_turn)

            # Persist the updated note
            self.storage.save_session_note(session_id, note)

            # Build transcript tail for context display
            tail = transcript_tail(note["turns"], n=5)

            # Rich response payload
            rich = {
                "agent_turn": agent_turn,
                "user_turn": user_turn,
                "transcript_tail": tail,
                "safety_banner": safety_payload,
                "display_delay_ms": display_delay_ms,
            }

            # Legacy-compatible fields
            legacy = {
                "session_id": session_id,
                "message": agent_turn["text"],
                "next_key": next_q["key"],
                "red_flags": red_flags,
                "redactions": redactions,
                "structured": note["structured"],
                "complete": False,
            }
            legacy.update(rich)  # include rich fields at top-level for client ease
            return legacy
        else:
            # Completion flow
            completion_text = "Thank you. I have collected your basic information."
            # Acknowledge then close
            agent_turn = {
                "from": "agent",
                "role": "assistant",
                "type": "completion",
                "text": f"{ack_text} {completion_text}",
                "key": None,
                "timestamp": now_iso(),
                "suggestions": None,
                "hints": None,
            }
            note["turns"].append(agent_turn)
            self.storage.save_session_note(session_id, note)
            tail = transcript_tail(note["turns"], n=5)

            rich = {
                "agent_turn": agent_turn,
                "user_turn": user_turn,
                "transcript_tail": tail,
                "safety_banner": safety_payload,
                "display_delay_ms": display_delay_ms,
            }
            legacy = {
                "session_id": session_id,
                "message": agent_turn["text"],
                "red_flags": red_flags,
                "redactions": redactions,
                "structured": note["structured"],
                "complete": True,
            }
            legacy.update(rich)
            return legacy


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
