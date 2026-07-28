

from loader import PDFLoader

loads = PDFLoader()
l1 = loads.load("data")
print(len(l1))
print(type(l1))
print(l1[0].metadata)
print(l1[0].page_content[:100])
from cleaner import TextCleaner
print ("------------------")
cl = TextCleaner()
c = cl.clean([l1[0]])

print(c)
from chunker import TextChunker

ch = TextChunker()
cq =ch.chunk([l1[0]])
print ("------------------")

print(cq)
