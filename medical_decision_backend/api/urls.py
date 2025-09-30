from django.urls import path
from .views import (
    health,
    chat_view,
    recommend_view,
    upload_report_view,
    get_recommendation_view,
)

urlpatterns = [
    path('health/', health, name='Health'),
    path('chat/', chat_view, name='Chat'),
    path('recommend/', recommend_view, name='Recommend'),
    path('upload_report/', upload_report_view, name='UploadReport'),
    path('get_recommendation/', get_recommendation_view, name='GetRecommendation'),
]
