"""
Django Admin Registration for Argus models
"""
from django.contrib import admin
from .models import Camera, Zone, Event, BehaviorProfile


@admin.register(Camera)
class CameraAdmin(admin.ModelAdmin):
    list_display = ['id', 'name', 'location_tag', 'status', 'fps']
    list_filter = ['status']
    search_fields = ['name', 'location_tag', 'rtsp_url']


@admin.register(Zone)
class ZoneAdmin(admin.ModelAdmin):
    list_display = ['id', 'name', 'camera', 'type']
    list_filter = ['type']
    search_fields = ['name']


@admin.register(Event)
class EventAdmin(admin.ModelAdmin):
    list_display = ['id', 'rule_type', 'camera', 'priority', 'status', 'timestamp']
    list_filter = ['priority', 'status', 'rule_type']
    search_fields = ['rule_type']


@admin.register(BehaviorProfile)
class BehaviorProfileAdmin(admin.ModelAdmin):
    list_display = ['person_id', 'created_at', 'updated_at']
    search_fields = ['person_id']