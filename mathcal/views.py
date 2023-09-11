from django.http import JsonResponse
from django.shortcuts import render
import json
# Create your views here.
from rest_framework import generics
from rest_framework.response import Response
from .models import Chapter, Formula, CalculationResult
from django.utils.decorators import method_decorator
from .serializers import ChapterSerializer, FormulaSerializer, CalculationResultSerializer

import numpy as np  # Import Numpy for matrix operations
import sympy as sp
import matplotlib
matplotlib.use('Agg')  # Set the Matplotlib backend to Agg before importing pyplot
import matplotlib.pyplot as plt
from io import BytesIO
import base64


class PlotFunctionView(generics.CreateAPIView):
    def post(self, request, *args, **kwargs):
        try:
            # Get the input function from the request
            input_function = request.data.get('function')
            print("INOUT FUNCTION ", input_function)
             # Generate a range of x values
            x_values = np.linspace(-1, 1, 3)
            # Evaluate the input function for each x value
            y_values = [eval(input_function) for x in x_values]
            curve, = plt.plot(x_values,y_values) 
            xdata = curve.get_xdata()
            ydata = curve.get_ydata()
            print("Extracting data from plot....")
            print("X data points for the plot is: ", xdata)
            print("Y data points for the plot is: ", ydata)
            # # Return the coordinates and the base64-encoded image
            data = {
                'x_values': xdata.tolist(),
                'y_values': ydata.tolist(),
                # 'plot_image': plot_base64,
            }

            return JsonResponse(data)
        except Exception as e:
            return JsonResponse({'error': str(e)})


class CalculationView(generics.CreateAPIView):
    queryset = CalculationResult.objects.all()
    serializer_class = CalculationResultSerializer

    def perform_create(self, serializer):
        formula_id = self.request.data.get('formula')
        input_values = self.request.data.get('input_values')
        formula = Formula.objects.get(pk=formula_id)

        if formula.name == 'Cross Product of Matrices':
            result = self.calculate_cross_product(input_values)
            # Convert the nested list result to JSON
            result_json = json.dumps(result)
            serializer.validated_data['result'] = result_json
        elif formula.name == 'Determinant of a Matrix':
            result = self.calculate_determinant(input_values)
            serializer.validated_data['result'] = result
        elif formula.name == 'Inverse of a Matrix':
            result = self.calculate_inverse(input_values)
            serializer.validated_data['result'] = result
        elif formula.name == 'Rank of a Matrix':
            result = self.calculate_rank(input_values)
            serializer.validated_data['result'] = result
        elif formula.name == 'Transpose of a Matrix':
            result = self.calculate_transpose(input_values)
            serializer.validated_data['result'] = result
        elif formula.name == 'Eigenvalues and Eigenvectors of a Matrix':
            result = self.calculate_eigenvalues(input_values)
            serializer.validated_data['result'] = result
        elif formula.name == 'Linear Dependence of Vectors':
            result = self.determine_linear_dependence(input_values)
            serializer.validated_data['result'] = result
        elif formula.name == 'Add Vectors':
            result = self.add_vectors(input_values)
            serializer.validated_data['result'] = result
        elif formula.name == 'Subtract Vectors':
            result = self.subtract_vectors(input_values)
            serializer.validated_data['result'] = result
        elif formula.name == 'Dot Product of Vectors':
            result = self.vector_dot_product(input_values)
            serializer.validated_data['result'] = result
        elif formula.name == 'Cross Product of Vectors':
            result = self.vector_cross_product(input_values)
            serializer.validated_data['result'] = result

        serializer.save(formula=formula, input_values=input_values)

    def calculate_cross_product(self, input_values):
        matrix1 = np.array(input_values['matrix1'])
        matrix2 = np.array(input_values['matrix2'])
        cross_product_result = np.cross(matrix1, matrix2)
        print('Cross Product ', cross_product_result)
        return cross_product_result.tolist()
    
    def calculate_determinant(self, input_values):
        matrix = np.array(input_values['matrix'])
        matrix = matrix.astype(float)
        determinant_result = np.linalg.det(matrix)
        return determinant_result
    def calculate_inverse(self, input_values):
        matrix = np.array(input_values['matrix'])
        matrix = matrix.astype(float)
        try:
            inverse_result = np.linalg.inv(matrix)
            return inverse_result.tolist()
        except np.linalg.LinAlgError:
            # Handle the case where the matrix is singular and has no inverse
            return "Matrix is singular and has no inverse"
        
    def calculate_rank(self, input_values):
        matrix = np.array(input_values['matrix'])
        matrix = matrix.astype(float)
        rank_result = np.linalg.matrix_rank(matrix)
        rank_result = int(rank_result)
        return rank_result
    
    def calculate_transpose(self, input_values):
        matrix = np.array(input_values['matrix'])
        matrix = matrix.astype(float)
        transpose_result = matrix.T.tolist()
        return transpose_result
    def calculate_eigenvalues(self, input_values):
        matrix = np.array(input_values['matrix'])
        print("INPUT matrix ", matrix)
        matrix = matrix.astype(int)
        print("INPUT matrix after ", matrix)
        w, v = np.linalg.eig(matrix)
        data = []
        eigenvalues = np.round(w, decimals=5)
        eigenvectors = np.round(v, decimals=5)
        print("Eigen values ", w.tolist())
        print("Eigen vectors", v.tolist())
        data.append({'x_values': eigenvalues.tolist()})
        data.append({'y_values': eigenvectors.tolist()})
        return data
    def determine_linear_dependence(self, input_values):
        matrix = np.array(input_values['matrix'])
        print("INPUT matrix ", matrix)
        matrix = matrix.astype(int)
        print("INPUT matrix after ", matrix)
        _, indexes = sp.Matrix(matrix).T.rref()  # T is for transpose
        print(indexes)
        print(matrix[indexes,:])
        linear_dependence = ""
        if len(indexes) == 2:
            print("linearly independant")
            linear_dependence = "linearly independant"
        else:
            print("linearly dependant")
            linear_dependence = "linearly dependant"
        data = []
        print("Input list  ", matrix.tolist())
        data.append({'input': matrix.tolist()})
        data.append({'dependence': linear_dependence})
        return data
    def add_vectors(self, input_values):
        matrix = np.array(input_values['matrix'])
        print("INPUT matrix ", matrix)
        matrix = matrix.astype(int)
        print("INPUT matrix after ", matrix)
        arr1 = matrix.tolist()[0]
        arr2 = matrix.tolist()[1]
        
        print ("1st array : ", arr1)  
        print ("2nd array : ", arr2)  
        
        out_arr = np.add(arr1, arr2)  
        data = []
        print("Input list  ", matrix.tolist())
        data.append({'input': matrix.tolist()})
        data.append({'sum': out_arr.tolist()})
        return data
    def subtract_vectors(self, input_values):
        matrix = np.array(input_values['matrix'])
        print("INPUT matrix ", matrix)
        matrix = matrix.astype(int)
        print("INPUT matrix after ", matrix)
        arr1 = matrix.tolist()[0]
        arr2 = matrix.tolist()[1]
        
        print ("1st array : ", arr1)  
        print ("2nd array : ", arr2)  
        
        out_arr = np.subtract(arr1, arr2)  
        data = []
        print("Input list  ", out_arr.tolist())
        data.append({'input': matrix.tolist()})
        data.append({'sum': out_arr.tolist()})
        return data
    def vector_dot_product(self, input_values):
        matrix = np.array(input_values['matrix'])
        print("INPUT matrix ", matrix)
        matrix = matrix.astype(int)
        print("INPUT matrix after ", matrix)
        arr1 = matrix.tolist()[0]
        arr2 = matrix.tolist()[1]
        
        print ("1st array : ", arr1)  
        print ("2nd array : ", arr2)  
        
        out_arr = np.dot(arr1, arr2)  
        data = []
        print("Input list  ", out_arr)
        data.append({'input': matrix.tolist()})
        data.append({'sum': out_arr.tolist()})
        return data
    def vector_cross_product(self, input_values):
        matrix = np.array(input_values['matrix'])
        print("INPUT matrix ", matrix)
        matrix = matrix.astype(int)
        print("INPUT matrix after ", matrix)
        arr1 = matrix.tolist()[0]
        arr2 = matrix.tolist()[1]
        
        print ("1st array : ", arr1)  
        print ("2nd array : ", arr2)  
        
        out_arr = np.cross(arr1, arr2)  
        data = []
        print("Input list  ", out_arr.tolist())
        data.append({'input': matrix.tolist()})
        data.append({'sum': out_arr.tolist()})
        return data