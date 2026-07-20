"""
Django Admin Registration for Argus models
"""
from django.contrib import admin
from django.db import models


class Camera(models.Model):
    name = models.CharField(max_length=255)
    location_tag = models.CharField(max_length=255, blank=True, null=True)
    rtsp_url = models.TextField(unique=True)
    status = models.CharField(max_length=50, default='offline')
    fps = models.FloatField(default=0.0)
    last_frame_time = models.DateTimeField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        app_label = 'argus_admin'

    def __str__(self):
        return self.name


class Zone(models.Model):
    camera = models.ForeignKey(Camera, on_delete=models.CASCADE, related_name='zones')
    name = models.CharField(max_length=255)
    type = models.CharField(max_length=50, default='polygon')
    coordinates = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        app_label = 'argus_admin'

    def __str__(self):
        return f"{self.name} ({self.camera.name})"


class Event(models.Model):
    camera = models.ForeignKey(Camera, on_delete=models.CASCADE, related_name='events')
    timestamp = models.DateTimeField()
    rule_type = models.CharField(max_length=50)
    object_type = models.CharField(max_length=50, blank=True, null=True)
    confidence = models.FloatField(blank=True, null=True)
    bbox = models.TextField(blank=True, null=True)
    snapshot_path = models.TextField(blank=True, null=True)
    priority = models.CharField(max_length=50, default='medium')
    status = models.CharField(max_length=50, default='new')
    metadata = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        app_label = 'argus_admin'
        ordering = ['-timestamp']

    def __str__(self):
        return f"{self.rule_type} - {self.camera.name}"


class BehaviorProfile(models.Model):
    person_id = models.CharField(max_length=255, unique=True)
    patterns = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        app_label = 'argus_admin'

    def __str__(self):
        return f"Profile: {self.person_id}"


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