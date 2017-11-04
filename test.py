from hyperlpr import pipline as pp
import cv2
import h5py

image = cv2.imread("6.jpg")
image, res = pp.SimpleRecognizePlate(image)
