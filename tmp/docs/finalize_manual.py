from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn


matches = list(Path("output/doc").glob("*.docx"))
assert len(matches) == 1, matches
path = matches[0]
document = Document(path)
settings = document.settings._element
for old in settings.findall(qn("w:updateFields")):
    settings.remove(old)
update = OxmlElement("w:updateFields")
update.set(qn("w:val"), "true")
settings.append(update)
document.save(path)
print(path)
