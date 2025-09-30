from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi

from .serializers import ChatRequestSerializer, RecommendRequestSerializer, UploadReportSerializer, GetRecommendationQuerySerializer
from .utils import generate_session_id, b64_to_bytes, OCEAN_THEME
from .storage import HybridStorage
from .rag import LocalVectorStore
from .agents import PatientAgent, ClinicalAgent

# initialize shared components
_storage = HybridStorage()
_vector_store = LocalVectorStore()
_patient_agent = PatientAgent(_storage)
_clinical_agent = ClinicalAgent(_storage, _vector_store)


@api_view(['GET'])
def health(request):
    """
    Health check endpoint.
    Returns simple message indicating the server is running.
    """
    return Response({"message": "Server is up!"})


# PUBLIC_INTERFACE
@swagger_auto_schema(
    method='post',
    operation_id='chat',
    operation_summary='Chat with PatientAgent',
    operation_description=(
        'Send a message to the PatientAgent to collect structured patient information. '
        'The agent responds with a conversational acknowledgment, optional safety banner if red-flags are detected, '
        'suggestion chips, hint text, and the next question. The response preserves legacy fields for backward compatibility.'
    ),
    request_body=ChatRequestSerializer,
    responses={
        200: openapi.Response(description="Chat response", examples={
            "application/json": {
                "session_id": "uuid",
                "message": "Thanks, I noted your cough. When did it start, and has it changed over time?",
                "next_key": "onset",
                "red_flags": [],
                "redactions": [],
                "structured": {"chief_complaint": "cough"},
                "complete": False,
                "agent_turn": {
                    "from": "agent",
                    "role": "assistant",
                    "type": "question",
                    "text": "Thanks, I noted your cough. When did it start, and has it changed over time?",
                    "key": "onset",
                    "suggestions": ["Today", "Yesterday", "Last week", "A month ago"],
                    "hints": "You can say 'yesterday', '3 days ago', or a date."
                },
                "user_turn": {
                    "from": "user",
                    "type": "message",
                    "text": "I have a cough",
                    "redactions": [],
                    "red_flags": []
                },
                "transcript_tail": [],
                "safety_banner": None,
                "display_delay_ms": 550,
                "theme": OCEAN_THEME
            }
        })
    },
    tags=["chat"]
)
@api_view(['POST'])
def chat_view(request):
    """
    Chat with PatientAgent.

    Parameters:
      - session_id (optional): If omitted, a new session is created.
      - message: User's input text.

    Returns:
      Rich chat payload with:
      - agent_turn: {from, role, type, text, key, suggestions, hints}
      - user_turn: {text, redactions, red_flags}
      - transcript_tail: previous 3-5 turns for context
      - safety_banner: safety notice when red flags are detected or null
      - display_delay_ms: UI typing simulation hint
      Plus legacy fields for backward compatibility: message, next_key, structured, red_flags, redactions, complete.
    """
    serializer = ChatRequestSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    session_id = serializer.validated_data.get("session_id") or generate_session_id()
    message = serializer.validated_data["message"]

    result = _patient_agent.handle_message(session_id, message)
    result["theme"] = OCEAN_THEME
    return Response(result, status=status.HTTP_200_OK)


# PUBLIC_INTERFACE
@swagger_auto_schema(
    method='post',
    operation_id='recommend',
    operation_summary='Generate clinical recommendation',
    operation_description='Use ClinicalAgent to generate a structured provisional recommendation using RAG over locally indexed guidelines.',
    request_body=RecommendRequestSerializer,
    responses={
        200: openapi.Response(description="Recommendation", examples={
            "application/json": {
                "summary": "Preliminary clinical considerations based on provided information.",
                "recommendations": ["..."],
                "guideline_support": [{"score": 0.8, "excerpt": "passage text", "source": "local"}],
                "safety": "Safety note..."
            }
        })
    },
    tags=["recommendations"]
)
@api_view(['POST'])
def recommend_view(request):
    """
    Generate clinical recommendation for a given session.
    Parameters:
      - session_id (required)
      - patient_notes (optional)
      - top_k (optional, default 3)
    Returns:
      - JSON structured recommendation with guideline support and safety disclaimer.
    """
    serializer = RecommendRequestSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    session_id = serializer.validated_data["session_id"]
    extra = serializer.validated_data.get("patient_notes", "")
    top_k = serializer.validated_data.get("top_k", 3)

    result = _clinical_agent.recommend(session_id, extra_notes=extra, top_k=top_k)
    return Response(result, status=status.HTTP_200_OK)


# PUBLIC_INTERFACE
@swagger_auto_schema(
    method='post',
    operation_id='upload_report',
    operation_summary='Upload a clinical report file',
    operation_description='Upload a base64-encoded clinical report and associate it to a session. Stored in OneDrive if enabled, otherwise locally.',
    request_body=UploadReportSerializer,
    responses={200: openapi.Response(description="Upload result", examples={"application/json": {"session_id": "uuid", "filename": "report.pdf", "storage_path": "/abs/path/or/onedrive://..."}})},
    tags=["files"]
)
@api_view(['POST'])
def upload_report_view(request):
    """
    Upload a clinical report file.
    Parameters:
      - session_id
      - filename
      - content_base64
    Returns:
      - storage path or identifier
    """
    serializer = UploadReportSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    session_id = serializer.validated_data["session_id"]
    filename = serializer.validated_data["filename"]
    content_b64 = serializer.validated_data["content_base64"]

    try:
        content = b64_to_bytes(content_b64)
    except Exception:
        return Response({"detail": "Invalid base64 content"}, status=status.HTTP_400_BAD_REQUEST)

    path = _storage.save_file(session_id, filename, content)
    # Append file upload event to notes
    _storage.save_session_note(session_id, {"type": "file_upload", "filename": filename, "path": path})

    return Response({"session_id": session_id, "filename": filename, "storage_path": path}, status=status.HTTP_200_OK)


# PUBLIC_INTERFACE
@swagger_auto_schema(
    method='get',
    operation_id='get_recommendation',
    operation_summary='Get latest recommendation for session',
    operation_description='Retrieve the last generated recommendation note for the session.',
    manual_parameters=[
        openapi.Parameter('session_id', openapi.IN_QUERY, description="Session identifier", type=openapi.TYPE_STRING, required=True),
    ],
    tags=["recommendations"]
)
@api_view(['GET'])
def get_recommendation_view(request):
    """
    Get latest recommendation for a session.
    Query params:
      - session_id
    Returns:
      - recommendation payload if available.
    """
    serializer = GetRecommendationQuerySerializer(data=request.query_params)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    session_id = serializer.validated_data["session_id"]
    note = _storage.get_latest_note(session_id) or {}
    if note.get("type") == "recommendation":
        return Response(note.get("result", {}), status=status.HTTP_200_OK)
    # If latest is not recommendation, return minimal info
    return Response({"detail": "No recommendation available yet for this session."}, status=status.HTTP_404_NOT_FOUND)
