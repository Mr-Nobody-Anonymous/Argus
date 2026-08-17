"""
Django models for Argus admin

These models map onto the *existing* SQLite tables created by
backend/database/db.py. Each model therefore declares an explicit
``db_table`` and ``managed = False`` so Django reads and writes the real
Argus data instead of creating a parallel set of django_admin_* tables.

Column names that differ from Django's defaults (e.g. ``camera`` ->
``camera_id``) are pinned with ``db_column``.
"""
from django.db import models


class Camera(models.Model):
    name = models.CharField(max_length=255)
    location_tag = models.CharField(max_length=255, blank=True, null=True)
    rtsp_url = models.TextField(unique=True)
    status = models.CharField(max_length=50, default='offline')
    fps = models.FloatField(default=0.0)
    last_frame_time = models.DateTimeField(blank=True, null=True)
    created_at = models.DateTimeField(blank=True, null=True)
    updated_at = models.DateTimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'cameras'
        verbose_name = 'Camera'
        verbose_name_plural = 'Cameras'

    def __str__(self):
        return self.name


class Zone(models.Model):
    camera = models.ForeignKey(Camera, on_delete=models.CASCADE, related_name='zones',
                               db_column='camera_id')
    name = models.CharField(max_length=255)
    type = models.CharField(max_length=50, default='polygon')
    coordinates = models.TextField()
    created_at = models.DateTimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'zones'
        verbose_name = 'Zone'
        verbose_name_plural = 'Zones'

    def __str__(self):
        return f"{self.name} ({self.camera.name})"


class Event(models.Model):
    camera = models.ForeignKey(Camera, on_delete=models.CASCADE, related_name='events',
                               db_column='camera_id')
    timestamp = models.DateTimeField()
    rule_type = models.CharField(max_length=50)
    object_type = models.CharField(max_length=50, blank=True, null=True)
    confidence = models.FloatField(blank=True, null=True)
    bbox = models.TextField(blank=True, null=True)
    snapshot_path = models.TextField(blank=True, null=True)
    priority = models.CharField(max_length=50, default='medium')
    status = models.CharField(max_length=50, default='new')
    metadata = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'events'
        ordering = ['-timestamp']
        verbose_name = 'Event'
        verbose_name_plural = 'Events'

    def __str__(self):
        return f"{self.rule_type} - {self.camera.name}"


class BehaviorProfile(models.Model):
    person_id = models.CharField(max_length=255, unique=True)
    patterns = models.TextField()
    created_at = models.DateTimeField(blank=True, null=True)
    updated_at = models.DateTimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'behavior_profiles'
        verbose_name = 'Behavior Profile'
        verbose_name_plural = 'Behavior Profiles'

    def __str__(self):
        return f"Profile: {self.person_id}"