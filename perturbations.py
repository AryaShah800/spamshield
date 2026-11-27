"""Spammer-style text obfuscations used by the robustness benchmark and the API's /probe endpoint."""

HOMOGLYPHS = str.maketrans({'a': 'а', 'e': 'е', 'o': 'о', 'p': 'р', 'c': 'с', 'x': 'х', 'i': 'і'})
LEET = str.maketrans({'a': '4', 'e': '3', 'i': '1', 'o': '0', 's': '5', 't': '7'})


def leetspeak(text):
    return text.translate(LEET)


def homoglyphs(text):
    return text.translate(HOMOGLYPHS)


def dotted_words(text):
    """Insert a dot between the letters of every word of 5+ letters (f.r.e.e style)."""
    return ' '.join('.'.join(w) if w.isalpha() and len(w) >= 5 else w for w in text.split(' '))


def spaced_digits_and_links(text):
    return text.replace('http', 'hxxp').replace('www.', 'www .').replace('.com', ' . com')


PERTURBATIONS = {'leetspeak': leetspeak, 'homoglyphs': homoglyphs, 'dotted_words': dotted_words,
                 'link_obfuscation': spaced_digits_and_links}
