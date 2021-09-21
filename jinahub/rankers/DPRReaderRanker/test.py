from jina import Document, Flow
from jina.types.request import Response


question1 = Document(text='When was Napoleon born?')
matches1 = [
    Document(
        text='Napoléon Bonaparte[a] (born Napoleone di Buonaparte; 15 August 1769 – 5'
        ' May 1821), usually referred to as simply Napoleon in English'
    ),
    Document(
        text='On 20 March 1811, Marie Louise gave birth to a baby boy, whom Napoleon'
        ' made heir apparent and bestowed the title of King of Rome.'
    ),
]
question1.matches.extend(matches1)

f = Flow().add(
    uses='jinahub+docker://DPRReaderRanker', uses_with={'num_spans_per_match': 1}
)


def print_matches(response: Response):
    for match in response.data.docs[0].matches:
        score = match.scores['relevance_score'].value
        print(f'Napoleon was born on {match.text} [relevance {score:.2%}]')


with f:
    f.post('/rank', question1, on_done=print_matches)
