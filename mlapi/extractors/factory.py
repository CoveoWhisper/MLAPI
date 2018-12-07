import requests
import hashlib
from .HTMLExtractor import HTMLExtractor
from .PDFExtractor import PDFExtractor

class ExtractorFactory:
    supportedFileTypes = set(['html', 'pdf'])
    htmlNumber = 0
    pdfNumber = 0
    
    def __init__(self):
        self.visitedFiles = set()
    
    def fromFilePath(self, path, fileType, headers = {}, preventAlreadySeenFiles = True):
        if fileType not in self.supportedFileTypes:
            return None

        try:
            response = requests.get(path, headers=headers, allow_redirects=True)
        except requests.exceptions.InvalidSchema as e:
            print("Invalid schema for Uri: " + path)
            return None
        except Exception as e:
            print("Exception happened for Uri: " + path)
            return None
        
        if response.status_code != 200:
            return None

        fileHash = hashlib.sha256(response.content).hexdigest()
        if fileHash in self.visitedFiles:
            return None
        self.visitedFiles.add(fileHash)
        
        if fileType == 'html':
            self.htmlNumber += 1
            if self.htmlNumber % 100 == 0:
                print(self.htmlNumber)
            return HTMLExtractor(response.content, response.encoding)
        elif fileType == 'pdf':
            localFilePath = str(self.pdfNumber)+'.pdf'
            self.pdfNumber += 1
            open(localFilePath, 'wb').write(response.content)
            return PDFExtractor(localFilePath)