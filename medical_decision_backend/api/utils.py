import base64
import json
import os
import re
import uuid
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

OCEAN_THEME = {
    "name": "Ocean Professional",
    "primary": "#2563EB",
    "secondary": "#F59E0B",
    "success": "#F59E0B",
    "error": "#EF4444",
    "gradient": "from-blue-500/10 to-gray-50",
    "background": "#f9fafb",
    "surface": "#ffffff",
    "text": "#111827",
}

# PUBLIC_INTERFACE
def generate_session_id() -> str:
    """Generate a unique session ID."""
    return str(uuid.uuid4())

# PUBLIC_INTERFACE
def redact_phi(text: str) -> Tuple[str, List[str]]:
    """
    Redact PHI from text using simple heuristic patterns.
    Returns redacted_text and list of redaction notes describing what was redacted.
    """
    if not text:
        return text, []

    redactions = []

    # Email addresses
    email_pattern = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    if email_pattern.search(text):
        redactions.append("Email addresses redacted")
    text = email_pattern.sub("[REDACTED_EMAIL]", text)

    # Phone numbers
    phone_pattern = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?){1,2}\d{4}")
    if phone_pattern.search(text):
        redactions.append("Phone numbers redacted")
    text = phone_pattern.sub("[REDACTED_PHONE]", text)

    # Dates (simple patterns)
    date_pattern = re.compile(r"\b(?:\d{1,2}[/.-]){2}\d{2,4}\b")
    if date_pattern.search(text):
        redactions.append("Dates redacted")
    text = date_pattern.sub("[REDACTED_DATE]", text)

    # Names (very naive: "Name: John Doe" or "Patient: John")
    name_pattern = re.compile(r"\b(?:Name|Patient|Pt|Mr|Mrs|Ms|Dr)\s*[:\-]\s*[A-Z][a-z]+(?:\s[A-Z][a-z]+)?")
    if name_pattern.search(text):
        redactions.append("Names redacted")
    text = name_pattern.sub("[REDACTED_NAME]", text)

    # Addresses (very naive)
    address_pattern = re.compile(r"\b\d{1,5}\s[A-Za-z0-9.\s]+(?:Street|St|Avenue|Ave|Road|Rd|Blvd|Lane|Ln|Way)\b", re.IGNORECASE)
    if address_pattern.search(text):
        redactions.append("Addresses redacted")
    text = address_pattern.sub("[REDACTED_ADDRESS]", text)

    return text, redactions

RED_FLAG_KEYWORDS = [
    "chest pain", "shortness of breath", "difficulty breathing", "bluish lips",
    "stroke", "facial droop", "slurred speech", "severe headache",
    "numbness on one side", "weakness on one side",
    "suicidal", "homicidal", "intent to harm", "hallucinations",
    "severe bleeding", "uncontrolled bleeding", "fainting", "syncope",
    "high fever", "stiff neck", "confusion", "seizure", "pregnant and bleeding",
    "anaphylaxis", "swelling of tongue", "cannot swallow", "allergic reaction"
]

# PUBLIC_INTERFACE
def detect_red_flags(text: str) -> List[str]:
    """Detect potential red-flag terms from free text."""
    found = []
    low = (text or "").lower()
    for k in RED_FLAG_KEYWORDS:
        if k in low:
            found.append(k)
    return found

# PUBLIC_INTERFACE
def b64_to_bytes(b64: str) -> bytes:
    """Decode base64 content safely."""
    return base64.b64decode(b64.encode("utf-8"))

# PUBLIC_INTERFACE
def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)

# PUBLIC_INTERFACE
def now_iso() -> str:
    """Return current UTC time in ISO format."""
    return datetime.utcnow().isoformat() + "Z"

# PUBLIC_INTERFACE
def theme_header_html(title: str) -> str:
    """
    Produce a minimal Ocean Professional themed header HTML snippet.
    This can be surfaced in HTML responses if needed.
    """
    return f"""
    <div style="background:{OCEAN_THEME['background']};padding:16px;border-bottom:1px solid #e5e7eb;">
      <div style="max-width:960px;margin:0 auto;">
        <h1 style="color:{OCEAN_THEME['primary']};margin:0;">{title}</h1>
        <p style="color:{OCEAN_THEME['text']};margin:4px 0 0 0;">Modern, minimal, and accessible.</p>
      </div>
    </div>
    """

# PUBLIC_INTERFACE
def safe_json_dump(data: Dict[str, Any]) -> str:
    """Dump JSON safely for storage."""
    return json.dumps(data, indent=2, ensure_ascii=False)

# -------------------- Conversational Helpers --------------------

DURATION_PAT = re.compile(r"\b(?:for\s*)?(\d+)\s*(?:day|days|d|week|weeks|w|month|months|m)\b", re.I)
SEVERITY_PAT = re.compile(r"\b(?:severity|pain|rate|score)?\s*(?:is|:)?\s*(\d{1,2})\b")
ONSET_PAT = re.compile(r"\b(?:started|onset|since)\s+(?:on\s+)?([A-Za-z]+|\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}|yesterday|today|last\s+night)\b", re.I)
ALLERGY_PAT = re.compile(r"\b(allerg(?:y|ies)|allergic to)\b", re.I)

# PUBLIC_INTERFACE
def extract_fields_from_text(text: str, expected_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract multiple structured fields from a single user message using simple regex heuristics.
    If expected_key is provided, it will bias the extraction by ensuring that specific field is preferred if ambiguous.

    Returns a dict subset of possible keys:
    - chief_complaint, onset, duration, severity, associated_symptoms, medical_history, medications, allergies
    """
    res: Dict[str, Any] = {}

    low = (text or "").lower().strip()

    # Chief complaint heuristic: short free-text if we are at that step.
    if expected_key == "chief_complaint" and low:
        res["chief_complaint"] = text.strip()

    # Duration
    m = DURATION_PAT.search(text or "")
    if m:
        val = m.group(0)
        res["duration"] = val

    # Severity (ensure 1-10 range if possible)
    ms = SEVERITY_PAT.findall(text or "")
    if ms:
        try:
            num = int(ms[-1])
            if 0 < num <= 10:
                res["severity"] = str(num)
        except Exception:
            pass

    # Onset
    mo = ONSET_PAT.search(text or "")
    if mo:
        res["onset"] = mo.group(0)

    # Allergies presence
    if ALLERGY_PAT.search(text or ""):
        # capture a simple phrase if present after "allergic to"
        after = re.search(r"allergic to\s+([A-Za-z0-9 ,]+)", low)
        res["allergies"] = after.group(1).strip() if after else "mentioned"

    # Associated symptoms cue
    if any(kw in low for kw in ["also", "other symptom", "as well as", "plus", "along with"]):
        res["associated_symptoms"] = text.strip()

    # Medications cue
    if any(kw in low for kw in ["taking", "dose", "mg", "tablet", "capsule", "medication", "medicine"]):
        res["medications"] = text.strip()

    # Medical history cue
    if any(kw in low for kw in ["history of", "hx of", "diagnosed with", "chronic", "past medical"]):
        res["medical_history"] = text.strip()

    # Bias toward expected_key if missing
    if expected_key and expected_key not in res and low:
        if expected_key in ["associated_symptoms", "medical_history", "medications", "allergies", "onset"] and len(low) > 0:
            res[expected_key] = text.strip()

    return res

# PUBLIC_INTERFACE
def build_acknowledgment(prev_key: Optional[str], user_text: str) -> str:
    """
    Generate a brief, empathetic acknowledgment referencing the previous key and user's input.
    """
    snippet = (user_text or "").strip()
    if len(snippet) > 80:
        snippet = snippet[:80].rstrip() + "..."
    if not prev_key:
        return "Thanks. I noted your message."
    templates = {
        "chief_complaint": "Thanks, I noted your main concern: '{x}'.",
        "onset": "Got it. I noted when it started: '{x}'.",
        "severity": "Thanks, I recorded the severity.",
        "associated_symptoms": "Thanks for sharing those associated symptoms.",
        "medical_history": "Thanks, I noted your medical history.",
        "medications": "Thanks, I captured your current medications.",
        "allergies": "Thanks, I recorded allergies.",
    }
    base = templates.get(prev_key, "Thanks, I noted that.")
    return base.replace("{x}", snippet)

# PUBLIC_INTERFACE
def get_suggestions_for_key(key: Optional[str]) -> List[str]:
    """Return suggestion chips for the next expected key."""
    if key == "onset":
        return ["Today", "Yesterday", "Last week", "A month ago"]
    if key == "severity":
        return ["2", "5", "7", "9"]
    if key == "associated_symptoms":
        return ["Fever", "Runny nose", "Headache", "Shortness of breath"]
    if key == "medical_history":
        return ["Hypertension", "Diabetes", "Asthma", "None"]
    if key == "medications":
        return ["Paracetamol 500 mg", "Ibuprofen 200 mg", "None"]
    if key == "allergies":
        return ["Penicillin", "Peanuts", "Latex", "None known"]
    if key == "chief_complaint":
        return ["Cough", "Chest pain", "Headache", "Fever"]
    return []

# PUBLIC_INTERFACE
def get_hints_for_key(key: Optional[str]) -> Optional[str]:
    """Provide gentle hint text for how to answer the next question."""
    hints = {
        "chief_complaint": "A short phrase like 'cough' or 'headache' works.",
        "onset": "You can say 'yesterday', '3 days ago', or a date.",
        "severity": "Please rate from 1 (mild) to 10 (worst).",
        "associated_symptoms": "List any other symptoms, separated by commas.",
        "medical_history": "Major conditions or surgeries, e.g. 'diabetes', 'asthma'.",
        "medications": "Include name and dose if you remember.",
        "allergies": "Any medication or food allergies?",
    }
    return hints.get(key)

# PUBLIC_INTERFACE
def build_safety_banner(red_flags: List[str]) -> Optional[Dict[str, Any]]:
    """
    Build a safety banner payload if red flags present.
    """
    if not red_flags:
        return None
    text = "Potential red-flag symptom detected: " + "; ".join(sorted(set(red_flags))) + "."
    note = "If you experience severe symptoms, please seek emergency care immediately."
    return {
        "level": "warning",
        "title": "Safety notice",
        "text": text,
        "note": note,
    }

# PUBLIC_INTERFACE
def transcript_tail(turns: List[Dict[str, Any]], n: int = 4) -> List[Dict[str, Any]]:
    """
    Return the last n turns for quick context display to the client.
    """
    return turns[-n:] if turns else []
