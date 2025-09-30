from rest_framework import serializers


# PUBLIC_INTERFACE
class ChatRequestSerializer(serializers.Serializer):
    """Serializer for chat messages to PatientAgent."""
    session_id = serializers.CharField(required=False, allow_blank=True, help_text="Session identifier; if omitted, a new session is created.")
    message = serializers.CharField(help_text="User's message to PatientAgent.")


# PUBLIC_INTERFACE
class RecommendRequestSerializer(serializers.Serializer):
    """Serializer for recommendation requests to ClinicalAgent."""
    session_id = serializers.CharField(help_text="Existing session identifier with collected patient data.")
    patient_notes = serializers.CharField(required=False, allow_blank=True, help_text="Optional free-text notes to enrich the context.")
    top_k = serializers.IntegerField(required=False, default=3, help_text="Number of guideline passages to retrieve via RAG.")


# PUBLIC_INTERFACE
class UploadReportSerializer(serializers.Serializer):
    """Serializer for uploading clinical reports."""
    session_id = serializers.CharField(help_text="Session identifier to associate the uploaded report.")
    filename = serializers.CharField(help_text="Original filename for the report. Used for storage and retrieval.")
    content_base64 = serializers.CharField(help_text="Base64-encoded file content.")


# PUBLIC_INTERFACE
class GetRecommendationQuerySerializer(serializers.Serializer):
    """Serializer for retrieving latest recommendation for a session."""
    session_id = serializers.CharField(help_text="Session identifier to fetch the last generated recommendation.")
