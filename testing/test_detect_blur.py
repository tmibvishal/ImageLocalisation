from imutils import paths
import cv2

def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()

img = cv2.imread("v2/image885.jpg", 1)
fg = variance_of_laplacian(img)
text = "Not Blurry"
if fg < 100:
		text = "Blurry"

print(text)
