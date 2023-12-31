# Generated by Django 4.2.5 on 2023-09-04 13:19

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Chapter',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
            ],
        ),
        migrations.CreateModel(
            name='Formula',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
                ('formula_text', models.CharField(max_length=200)),
                ('chapter', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='mathcal.chapter')),
            ],
        ),
        migrations.CreateModel(
            name='CalculationResult',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('input_values', models.JSONField()),
                ('result', models.FloatField()),
                ('formula', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='mathcal.formula')),
            ],
        ),
    ]
