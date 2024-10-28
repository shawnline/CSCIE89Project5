import gensim
import nltk

smallTextBytes: bytes = open("TeachingComputerScienceBrief", "rb").read()
largeTextBytes: bytes = open("TeachingComputerScienceFull", "rb").read()
# Remove non ascii characters
smallText: str = smallTextBytes.decode("ascii", "replace").lower()
largeText: str = largeTextBytes.decode("ascii", "replace").lower()
# Re encode as ASCII so each character maps to one byte
smallTextCharMap: list[int] = list(smallText.encode("ascii", "replace"))
largeTextCharMap: list[int] = list(largeText.encode("ascii", "replace"))

print(smallTextCharMap[:100])