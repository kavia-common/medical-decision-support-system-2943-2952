import base64
import json
import os
import re
import uuid
from datetime import datetime
from typing import Dict, Any, List, Tuple


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
