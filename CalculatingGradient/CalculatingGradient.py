import cv2
from skimage import io

# Read image
src = cv2.imread("Maak1.png")
# Apply gaussian blur
cv2.GaussianBlur(src, (3, 3), 10)
# Convert image to grayscale
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
# Apply Sobel method to the grayscale image
grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
# Horizontal Sobel Derivation
grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
# Vertical Sobel Derivation
abs_grad_x = cv2.convertScaleAbs(grad_x)
abs_grad_y = cv2.convertScaleAbs(grad_y)
grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
# Apply both
# Show the image
cv2.imshow("Original Image", src)
cv2.imshow("Gradient Image", grad)
# save result image on disk
io.imsave("Results/Original Image.png", src)
io.imsave("Results/Gradient Image.png", grad)
# wait for key to end
cv2.waitKey(0)



