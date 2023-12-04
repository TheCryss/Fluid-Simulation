# NO USAR LIBRERIAS QUE CALCULEN LOS SPLINES, HAY QUE IMPLEMENTAR

import numpy as np
#from auxiliares import limpiar_pantalla
from prettytable import PrettyTable 
import matplotlib.pyplot as plt

# Se define la malla a tratar con las condiciones iniciales

v_0 = 1 # Velocidad inicial en x

mallax = np.array(
    [[1,  0,   0,   0,   0,   0,   0,  0],
    [1, "w1", "w2", 0, 0, "w3", "w4", 0],
    [1, "w5", "w6", 0, 0, "w7", "w8", 0],
    [1, "w9", "w10", "w11", "w12", "w13", "w14", 0],
    [1, "w15", "w16", 0, 0, "w17", "w18", 0],
    [1, "w19", "w20", 0, 0, "w21", "w22", 0],
    [1,  0,   0,   0,   0,   0,   0,  0]])

mallay = np.array(
    [[0,  0,   0,   0,   0,   0,   0,  0],
    [0, "w1", "w2", 0, 0, "w3", "w4", 0],
    [0, "w5", "w6", 0, 0, "w7", "w8", 0],
    [0, "w9", "w10", "w11", "w12", "w13", "w14", 0],
    [0, "w15", "w16", 0, 0, "w17", "w18", 0],
    [0, "w19", "w20", 0, 0, "w21", "w22", 0],
    [0,  0,   0,   0,   0,   0,   0,  0]])


r_matrix = np.zeros((22, 23), dtype=int) # Almacenar los resultados finales de la discretización

# Funtion that reshape results in a finish matrix the last colum are the numeric values

def fixMatrix(values, count):
    global r_matrix
    acum = 0
    for value in values:
        if value[0] == 'w':
            match np.where(values == value)[0][0]:
                case 0:
                    r_matrix[count][int(value[1:])-1] = 1
                case 1:
                    r_matrix[count][int(value[1:])-1] = -8
                case 2:
                    r_matrix[count][int(value[1:])-1] = 3
                case 3:
                    r_matrix[count][int(value[1:])-1] = 3
                case 4:
                    r_matrix[count][int(value[1:])-1] = 1
        else:
            match np.where(values == value)[0][0]:
                case 0:
                    acum += int(value)*1
                case 1:
                    acum += int(value)*-8
                case 2:
                    acum += int(value)*3
                case 3:
                    acum += int(value)*3
                case 4:
                    acum += int(value)*1
    r_matrix[count][-1] = acum


# Funtion that find the positions needed in navier-stokes using Finite differences
def discretization(malla):
    count = 0
    for row in range(7):
        for colum in range(8):
            if malla[row][colum][0] == 'w':
                temp = np.array([malla[row][colum+1], malla[row][colum],
                                malla[row][colum-1], malla[row-1][colum], malla[row+1][colum]])
                fixMatrix(temp, count)
                count += 1

discretization(mallax)

column_names = [f"W{i}" for i in range(1, 24)]
column_names[-1] = "T.I"

# Create a table to format de r_matrix table
table = PrettyTable()
table.field_names = [""] + column_names 

for i, row in enumerate(r_matrix, start=1):
    equation_label = f"Ecuación {i}"
    row_as_strings = [str(value) for value in row]
    table.add_row([equation_label] + row_as_strings)

'''
Funtion that checks if a matrix is diagonally dominant
X = Matrix
'''
def dd(X):
    M = np.delete(X,-1,1)
    D = np.diag(np.abs(M)) # Find diagonal coefficients
    S = np.sum(np.abs(M), axis=1) - D # Find row sum without diagonal
    if np.all(D >= S):
        print( 'matrix is diagonally dominant')
    else:
        print ('NOT diagonally dominant')

dd(r_matrix)

'''
Calculate the determinant of the matrix
'''
r_matrix_ni = np.delete(r_matrix,-1,1)

determinant = np.linalg.det(r_matrix_ni)
if determinant !=0:
    print("The matrix is invertible")
else:
    print("The matrix is not invertible")


'''
Funtion that use jacobi method in a especific row
R = Row to apply jacobi
S = array of initial solutions for the row
nn = interation value, in other words for which variable we want to apply jacobi.
'''
def jacobi(R,S,nn):
    acum =0
    c = -R[-1]/R[nn]
   # print(f"{-R[-1]}/{R[nn]} = {c}")
    acum += c
    for i in range(len(R)-1):
        #print(f"{-R[i]} / {R[nn]}*{S[i]} = {(-R[i]/ R[nn]) * S[i]}")
        acum+= ( (-R[i]/ R[nn]) * S[i] ) if i != nn else 0
    return acum

'''
Funtion that checks if the given solutions satisfy the tolerance
S = Preview solution
NS = New Solution
tol = tolerance
'''
def check_tolerance(S, NS, tol):
    # Infinity Norm
    error = np.linalg.norm(NS - S, np.inf) / np.linalg.norm(NS, np.inf)
    print("Error: ", error)
    if error > tol:
        return True
    else:
        return False
    
    
    
'''
Funtion that use Successive Over-Relaxation in an especific matrix.
M = Matrix of coeficients with idependent vector  
omega = stride for faster convergence.
tol = tolerance
'''

def SOR(M,omega,tol):
    l = len(M)
    S= np.zeros(l)
    OS = np.copy(S)
    acum = 0
    for i in range(l):
        c = -M[i][-1]/M[i][i]
        S[i] = c    
    while check_tolerance(OS,S,tol):
        acum += 1
        OS = np.copy(S)
        for i in range(l):
            S[i] = (1 - omega) * S[i] + omega * jacobi(M[i],S,i)
    print(f"Number of iterations: {acum}")
    return S

lineal_solution = SOR(r_matrix,1.2,1e-6)
print("---------------")
# print(lineal_solution)

'''
line_solution: [0.49452272 0.2392135  0.000871   0.00094926 0.71696823 0.43013979 0.00601873 0.00498102 0.82203822 0.57257314 0.22593952 0.0897968 0.04055587 0.01894422 0.85282788 0.59811168 0.01805621 0.01494307 0.75839677 0.50869067 0.00783899 0.00854327]
'''

'''
The following MATLAB M-function demonstrates this algorithm in action:
% GBI.M
% Written December 2009, (C) Matthew Giassa
% teo@giassa.net, www.giassa.net
% Returns an upsampled image using bicubic interpolation
function output_image = gbi( input_image,x_res,y_res )
%   input_image     -   an image on which to perform bicubic interpolation
%   x_res           -   the new horizontal dimensions (in pixels)
%   y_res           -   the new vertical dimensions (in pixels)
%Define the inverted weighting matrix, M^(-1), no need to recalculate it
%ever again
M_inv = [
 1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
 0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0;
 -3,3,0,0,-2,-1,0,0,0,0,0,0,0,0,0,0;
 2,-2,0,0,1,1,0,0,0,0,0,0,0,0,0,0;
 0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0;
 0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0;
 0,0,0,0,0,0,0,0,-3,3,0,0,-2,-1,0,0;
 0,0,0,0,0,0,0,0,2,-2,0,0,1,1,0,0;
 -3,0,3,0,0,0,0,0,-2,0,-1,0,0,0,0,0;
 0,0,0,0,-3,0,3,0,0,0,0,0,-2,0,-1,0;
 9,-9,-9,9,6,3,-6,-3,6,-6,3,-3,4,2,2,1;
 -6,6,6,-6,-3,-3,3,3,-4,4,-2,2,-2,-2,-1,-1;
 2,0,-2,0,0,0,0,0,1,0,1,0,0,0,0,0;
 0,0,0,0,2,0,-2,0,0,0,0,0,1,0,1,0;
 -6,6,6,-6,-4,-2,4,2,-3,3,-3,3,-2,-1,-2,-1;
 4,-4,-4,4,2,2,-2,-2,2,-2,2,-2,1,1,1,1
 ];
%Make a copy of the input image
I = input_image;
%Convert image to grayscale (intensity) values for simplicity (for now)
I = double(rgb2gray(I));
%Determine the dimensions of the source image
%Note that we will have three values - width, height, and the number
%of color vectors, 3
[j k] = size(I);
%Specify the new image dimensions we want for our larger output image
x_new = x_res;
y_new = y_res;
%Determine the ratio of the old dimensions compared to the new dimensions
%Referred to as S1 and S2 in my tutorial
x_scale = x_new./(j-1);
y_scale = y_new./(k-1);
%Declare and initialize an output image buffer
temp_image = zeros(x_new,y_new);
%Calculate the horizontal derivatives for each point (assume that at the
%edge pixels of the source image, all derivatives are zero to maximize
%smoothing). Take note - recreating this in C or C++ might not be so easy,
%as you must pay attention to how this code is using matrix-style
%addressing (x-y positions swapped).
Ix = double(zeros(j,k));
for count1 = 1:j
 for count2 = 1:k
 if( (count2==1) || (count2==k) )
 Ix(count1,count2)=0;
 else
 Ix(count1,count2)=(0.5).*(I(count1,count2+1)-I(count1,count2-1));
 end
 end
end
%Similarly, calculate the vertical derivatives
Iy = double(zeros(j,k));
for count1 = 1:j
 for count2 = 1:k
 if( (count1==1) || (count1==j) )
 Iy(count1,count2)=0;
 else
 Iy(count1,count2)=(0.5).*(I(count1+1,count2)-I(count1-1,count2));
 end
 end
end
%Finally, calculate the cross derivatives
Ixy = double(zeros(j,k));
for count1 = 1:j
 for count2 = 1:k
 if( (count1==1) || (count1==j) || (count2==1) || (count2==k) )
 Ixy(count1,count2)=0;
 else
 Ixy(count1,count2)=(0.25).*((I(count1+1,count2+1)+I(count1-1,count2-1)) - (I(count1+1,count2-1)+I(count1-1,count2+1)));
 end
 end
end
%Generate the output image
%==========================================================================
%Please note that this is definitely not the most efficient way to carry
%out this operation, as we are recalculating our beta and alpha-vectors far
%repeatedly when there is no need. I kept the code like this so you can see
%the order in which we would carry out most operations. If you want to make
%a more efficient version which pre-calculates the alpha vector for each
%pixel in the source image, be my guest.
%==========================================================================
for count1 = 0:x_new-1
 for count2 = 0:y_new-1
 %Calculate the normalized distance constants, h and w
 W = -(((count1./x_scale)-floor(count1./x_scale))-1);
 H = -(((count2./y_scale)-floor(count2./y_scale))-1);
 %Determine the indexes/address of the 4 neighbouring pixels from
 %the source data/image
 I11_index = [1+floor(count1./x_scale),1+floor(count2./y_scale)];
 I21_index = [1+floor(count1./x_scale),1+ceil(count2./y_scale)];
 I12_index = [1+ceil(count1./x_scale),1+floor(count2./y_scale)];
 I22_index = [1+ceil(count1./x_scale),1+ceil(count2./y_scale)];
 %Calculate the four nearest function values
 I11 = I(I11_index(1),I11_index(2));
 I21 = I(I21_index(1),I21_index(2));
 I12 = I(I12_index(1),I12_index(2));
 I22 = I(I22_index(1),I22_index(2));
 %Calculate the four nearest horizontal derivatives
 Ix11 = Ix(I11_index(1),I11_index(2));
 Ix21 = Ix(I21_index(1),I21_index(2));
 Ix12 = Ix(I12_index(1),I12_index(2));
 Ix22 = Ix(I22_index(1),I22_index(2));
 %Calculate the four nearest vertical derivatives
 Iy11 = Iy(I11_index(1),I11_index(2));
 Iy21 = Iy(I21_index(1),I21_index(2));
 Iy12 = Iy(I12_index(1),I12_index(2));
 Iy22 = Iy(I22_index(1),I22_index(2));
 %Calculate the four nearest cross derivatives
 Ixy11 = Ixy(I11_index(1),I11_index(2));
 Ixy21 = Ixy(I21_index(1),I21_index(2));
 Ixy12 = Ixy(I12_index(1),I12_index(2));
 Ixy22 = Ixy(I22_index(1),I22_index(2));
 %Create our beta-vector
 beta = [I11 I21 I12 I22 Ix11 Ix21 Ix12 Ix22 Iy11 Iy21 Iy12 Iy22 Ixy11 Ixy21 Ixy12 Ixy22];
 %Calculate our alpha vector (ie: the a-values) using simple matrix
 %multiplication
 %==================================================================
 %If we wanted to make sure this entire program is written entirely
 %from scratch, we would implement our own code to multiple an AxB
 %matrix and a BxC matrix right here. This is very simple to do, so
 %I won't go over it right now.
 %==================================================================
 alpha = M_inv*beta';
 temp_p=0;
 for count3 = 1:16
 w_temp = floor((count3-1)/4);
 h_temp = mod(count3-1,4);
 %disp(sprintf('aij=%d  wsub=%d   hsub=%d\n',count3,w_temp,h_temp))
 temp_p = temp_p + alpha(count3).*((1-W)^(w_temp)).*((1-H)^(h_temp));
 end
 temp_image(count1+1,count2+1)=temp_p;
 end
end
output_image = temp_image;
'''

'''
TALLER
1.Calcule splines bi-cúbicos para modelar la solución obtenida con el método de Jacobí con sobre-relajación para los sistemas de ecuaciones lineales obtenidos con diferencias finitas al discretizar las componentes del fluido en el eje x (punto 1 de la entrega 2 del proyecto).

2.Genere una nueva gráfica para la solución obtenida con el método de Jacobí con sobre- relajación para la componente del fluido en el eje x con al menos el doble de filas y columnas de las utilizadas en la malla de discretización del problema (ver figura 2), los valores faltantes serán interpolados usando los splines bi-cúbicos calculados en el punto anterior
'''

# La funcion dada una matriz discretizada matriz, un valor inicial
# un valor de velocidad inicial v0, tolerancia TOL y un valor de omega W calcula
# los valores correspondientes en cada punto de la matriz de acuerdo
# a los valores de entrada.
def initial_solve_matrix(matriz, v0, TOL,W):
    dim = matriz.shape
    
    # Obtiene la matriz de coeficientes con la que se calculan los valores
    # incognita de la matriz
    discrete_matrix = discrete(matriz,fun,22,v0)
    
    # Calcula los valores incognita de la matriz dada las ecuaciones, usando el
    # metodo de Gauss-Seidel.
    sis_eqv = genSistemaEqv(discrete_matrix)
    values_Gauss_Seidel = solveGauss_Seidel(sis_eqv,TOL,W)
    
    m = []
    actual_value = 0

    # Reemplaza los valores calculados en la matriz, de manera que se obtiene
    # la matriz con los valores.
    for i in range(dim[0]):
        row = []
        for j in range(dim[1]):
            if(matriz[i,j][0] == "V"):
                row.append(v0)
            elif(matriz[i,j][0] == "W"):
                row.append(values_Gauss_Seidel[actual_value])
                actual_value += 1
            else:
                row.append(0)
        m.append(row)

    return np.array(m)

print("Matriz de valores")

# print("Matriz de valores")
# TOL = 0.0001
# W = 1.2
# matrix_values = initial_solve_matrix(mallax,10,TOL,W)