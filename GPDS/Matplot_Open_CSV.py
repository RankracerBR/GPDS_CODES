#Libs
import matplotlib.pyplot as plt
import cv2
import copy


#Variables
#Graphics
data = [-3,-2,-1,0,1,2,3]
print(data)

plt.stem(data)
plt.show()

data2 = [-20,-19,-18,-17,-16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
plt.stem(data2)
plt.show()

#Cv2
image_example = cv2.imread('images/bmw.jpg')
cv2.imshow("BMW",image_example)
cv2.waitKey(0)

print('Dados da imagem colorida:')
print(image_example.shape)

image_gray_example = cv2.imread("images/bmw.jpg", 0)
cv2.imshow("Gray BMW",image_gray_example)
cv2.waitKey(0)

print("Dados da imagem em tons de cinza: ")
print(image_gray_example.shape)


image_2_example = cv2.imread("images/renault_megane_st.jpeg")
image_show = cv2.imshow("Renault Megane", image_2_example)
cv2.waitKey(0)

image_2_example_blue = copy.copy(image_2_example)

image_2_example_blue[:,:,1] = 0
image_2_example_blue[:,:,2] = 0
cv2.imshow("Blue Megane",image_2_example_blue)
cv2.waitKey(0)

image_2_example_green = copy.copy(image_2_example)

image_2_example_green[:,:,0] = 0
image_2_example_green[:,:,2] = 0
cv2.imshow("Green Megane",image_2_example_green)
cv2.waitKey(0)

image_2_example_red = copy.copy(image_2_example)

image_2_example_red[:,:,0] = 0
image_2_example_red[:,:,1] = 0
cv2.imshow("Red Megane",image_2_example_red)
cv2.waitKey(0)

image_3_example_jimny = cv2.imread("images/suzuki_jimny.jpg")
cv2.imshow("Suzuki Jimny", image_3_example_jimny)
cv2.waitKey(0)

image_3_example_convertida_jimny = cv2.cvtColor(image_3_example_jimny, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray Suzuki Jimny",image_3_example_convertida_jimny)
cv2.waitKey(0)

image_example_jimny_convertida_HSV = cv2.cvtColor(image_3_example_jimny, cv2.COLOR_BGR2HSV)
cv2.imshow("HSV Suzuki Jimny", image_example_jimny_convertida_HSV)
cv2.waitKey(0)

image_example_jimny_convertida_RGB = cv2.cvtColor(image_3_example_jimny, cv2.COLOR_BGR2RGB)
cv2.imshow("RGB Suzuki Jimny",image_example_jimny_convertida_RGB)
cv2.waitKey(0)

image_example_megane_convertida_RGB = cv2.cvtColor(image_2_example, cv2.COLOR_BGR2RGB)
cv2.imshow("RGB Renault Megane",image_example_megane_convertida_RGB)
cv2.waitKey(0)

image_Sobel_example_Gray = cv2.cvtColor(image_2_example, cv2.COLOR_BGR2GRAY)
print('Imagem Original')
cv2.imshow("Renault Megane",image_Sobel_example_Gray)
print('\n\n')
cv2.waitKey(0)

gX = cv2.Sobel(image_Sobel_example_Gray, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3) #ksize: tamanho da matriz de vizinhaça dos bits
gY = cv2.Sobel(image_Sobel_example_Gray, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3) #o ddepth converte a imagem para outro formato

#Converter a imagem novamente para o formato 8 bits
gX = cv2.convertScaleAbs(gX)
gY = cv2.convertScaleAbs(gY)

#Juntando os gradientes em X e Y com o mesmo grau de importância
Sobel_grads_combined = cv2.addWeighted(gX, 0.5, gY, -.5, 0)

cv2.imshow("Gradiente de Sobel em X", gX)
cv2.waitKey(0)
cv2.imshow("Gradiente de Sobel em Y", gY)
cv2.waitKey(0)
cv2.imshow("Gradientes de Sobel em X e Y combinados", Sobel_grads_combined)
cv2.waitKey(0)

image_Canny_example_Gray = cv2.cvtColor(image_2_example, cv2.COLOR_BGR2GRAY)

Canny_1 = cv2.Canny(image_Canny_example_Gray, 10,50)
Canny_2 = cv2.Canny(image_Canny_example_Gray, 150,400)

cv2.imshow("Imagem Original", image_Canny_example_Gray)
cv2.waitKey(0)
cv2.imshow("Imagem 1 por Canny", Canny_1)
cv2.waitKey(0)
cv2.imshow("Imagem 2 por Canny", Canny_2)
cv2.waitKey(0)

image_Laplaciano_example_Gray = cv2.cvtColor(image_2_example, cv2.COLOR_BGR2GRAY)
cv2.imshow("Imagem original", image_Laplaciano_example_Gray)
cv2.waitKey(0)

laplaciano_image = cv2.Laplacian(image_Laplaciano_example_Gray, cv2.CV_64F, ksize=3)

laplaciano_image_abs = cv2.convertScaleAbs(laplaciano_image)

cv2.imshow("Imagem após a aplicação do laplaciano", laplaciano_image_abs)
cv2.waitKey(0)