# Generated by Django 5.1.1 on 2024-11-14 05:29

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name="BodyMeasurement",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("image", models.ImageField(upload_to="measurements/")),
                (
                    "processed_image",
                    models.ImageField(null=True, upload_to="processed/"),
                ),
                ("shoulder_width", models.FloatField(null=True)),
                ("chest_circumference", models.FloatField(null=True)),
                ("waist_circumference", models.FloatField(null=True)),
                ("hip_circumference", models.FloatField(null=True)),
                ("left_bicep_circumference", models.FloatField(null=True)),
                ("right_bicep_circumference", models.FloatField(null=True)),
                ("left_forearm_circumference", models.FloatField(null=True)),
                ("right_forearm_circumference", models.FloatField(null=True)),
                ("left_thigh_circumference", models.FloatField(null=True)),
                ("right_thigh_circumference", models.FloatField(null=True)),
                ("left_calf_circumference", models.FloatField(null=True)),
                ("right_calf_circumference", models.FloatField(null=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                (
                    "user",
                    models.ForeignKey(
                        null=True,
                        on_delete=django.db.models.deletion.CASCADE,
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
        ),
    ]