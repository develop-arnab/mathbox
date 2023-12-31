# Generated by Django 4.1 on 2023-10-04 19:51

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('mathcal', '0003_alter_calculationresult_result'),
    ]

    operations = [
        migrations.CreateModel(
            name='Question',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('question_text', models.CharField(max_length=200)),
                ('choice_1', models.CharField(max_length=200)),
                ('choice_2', models.CharField(max_length=200)),
                ('choice_3', models.CharField(max_length=200)),
                ('choice_4', models.CharField(max_length=200)),
                ('correct_choice', models.PositiveSmallIntegerField(choices=[(1, 'Choice 1'), (2, 'Choice 2'), (3, 'Choice 3'), (4, 'Choice 4')])),
            ],
        ),
    ]
