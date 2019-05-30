import pytesseract
txt=pytesseract.image_to_string("testResults/v1/image207.jpg")
print(txt)