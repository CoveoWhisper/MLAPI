import os
import subprocess

class PDFExtractor:
    def __init__(self, path):
        self.path = path

    def extractAllText(self):
        outputPath = self.path + '.txt'
        subprocess.call(['pdftotext', self.path, outputPath])
        with open(outputPath, 'r') as myFile:
            data = myFile.read()
        os.remove(outputPath)
        return data
