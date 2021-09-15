from jina import Document, Flow

from ...sentencizer import Sentencizer


def test_exec():
    f = Flow().add(uses=Sentencizer)
    with f:
        resp = f.post(
            on='/test',
            inputs=Document(text='Hello. World! Go? Back'),
            return_results=True,
        )
        assert resp[0].docs[0].chunks[0].text == 'Hello.'
        assert resp[0].docs[0].chunks[1].text == 'World!'
        assert resp[0].docs[0].chunks[2].text == 'Go?'
        assert resp[0].docs[0].chunks[3].text == 'Back'
