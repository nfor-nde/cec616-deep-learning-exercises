
def reverse(s):
    return s[::-1]

def is_palindrome(s):
    clean = s.lower().replace(" ", "")
    return clean == clean[::-1]

def word_count(s):
    return len(s.split())
