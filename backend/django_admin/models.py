"""
Django models for Argus admin
These models mirror the SQLite database for admin access
"""
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

    def __str__(self):
        return self.name


class Zone(models.Model):
    camera = models.ForeignKey(Camera, on_delete=models.CASCADE, related_name='zones')
    name = models.CharField(max_length=255)
    type = models.CharField(max_length=50, default='polygon')
    coordinates = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

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
        ordering = ['-timestamp']

    def __str__(self):
        return f"{self.rule_type} - {self.camera.name}"


class BehaviorProfile(models.Model):
    person_id = models.CharField(max_length=255, unique=True)
    patterns = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Profile: {self.person_id}"