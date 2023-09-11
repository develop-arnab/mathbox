from django.db import models

# Create your models here.

class Chapter(models.Model):
    name = models.CharField(max_length=100)

class Formula(models.Model):
    chapter = models.ForeignKey(Chapter, on_delete=models.CASCADE)
    name = models.CharField(max_length=100)
    formula_text = models.CharField(max_length=200)

class CalculationResult(models.Model):
    formula = models.ForeignKey(Formula, on_delete=models.CASCADE)
    input_values = models.JSONField()
    result = models.JSONField(null=True, blank=True)