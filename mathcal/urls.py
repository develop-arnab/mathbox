from django.urls import path
from .views import CalculationView, PlotFunctionView

urlpatterns = [
    # path('chapters/', ChapterList.as_view(), name='chapter-list'),
    # path('formulas/', FormulaList.as_view(), name='formula-list'),
    # path('results/', CalculationResultList.as_view(), name='result-list'),
    path('calculate/', CalculationView.as_view(), name='calculate'),
     path('plot_function/', PlotFunctionView.as_view(), name='plot_function'),
]