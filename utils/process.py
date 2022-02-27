import re

def partition(self, text):
    """将未切分的question进行切分，同时保留数字连一起"""
    process_text = []
    text_len = len(text)
    for i, char in enumerate(text):
        if i == (text_len - 1):
            process_text.append(char)
            continue
        if not self.is_number(char) and not self.is_alphabet(char) and char not in ['.', '%']:
            process_text.append(char + ' ')
        elif self.is_alphabet(char):
            next_char = text[i + 1]
            if self.is_alphabet(next_char):
                process_text.append(char)
            else:
                process_text.append(char + ' ')
        else:
            next_char = text[i + 1]
            if self.is_number(next_char) or next_char in ['.', '%']:
                process_text.append(char)
                continue
            else:
                process_text.append(char + ' ')
    question = ''.join(process_text).strip()
    question = re.sub('(\d+)/(\d+)', '\\1 / \\2', question)  # 1/5 -> 1 / 5
    question = re.sub('(\d+)\.(\d+)', '\\1 . \\2', question)  # 1.5 -> 1 . 5 [通过小数点进行切分]
    return question
