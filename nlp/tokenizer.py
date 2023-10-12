import re

class Tokenize():

    def __init__(self, text):
        self.text = text

    def byWords(self):
        # rpboelm with minus
        tokens = re.findall("[\w]+", self.text)
        return tokens

    def bySentence(self):
        """
        doesnt work by now
        """
        tokens = re.findall('[.!?] ', self.text)
        return tokens





def test():

    text = """this is a sentence with many wo-rds. so lets try. this. yo."""
    t = Tokenize(text)
    tokens = t.byWords()
    print(tokens)
    tokens = t.bySentence()
    print(tokens)

def main():
    test()


if __name__ == "__main__":
    main()
