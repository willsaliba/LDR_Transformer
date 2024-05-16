import re
import regex

test_string = "Here's a test, with 123 numbers and ll, re, ve patterns. Non-ASCII letters: ñ, é, ü."

pattern_rgx = regex.compile(r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""")
splits_rgx = regex.findall(pattern_rgx, test_string)

# r"(?i)[sdmt]|ll|ve|re|[^\r\n\w]?+[a-zA-Z]+|\d{1,3}|\s?[^\s\w]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"

pattern_re = r"(?i)[sdmt]|ll|ve|re|[^\r\n\w]?+[a-zA-Z]+|\d{1,3}|\s?[^\s\w]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"
compiled_re = re.compile(pattern_re)
splits_re = re.findall(compiled_re, test_string)

print(splits_re)
print("\n")
print(splits_rgx)