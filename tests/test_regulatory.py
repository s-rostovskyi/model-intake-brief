from __future__ import annotations

from intake import regulatory


class FakeResponse:
    def __init__(self, payload: dict[str, object], status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f'HTTP {self.status_code}')

    def json(self) -> dict[str, object]:
        return self._payload


def test_corpus_manifest_includes_verified_transparency_measures():
    manifest = regulatory.corpus_manifest()

    assert manifest['corpus_version'] == regulatory.CORPUS_VERSION
    assert manifest['embedding_model'] == regulatory.OPENAI_EMBEDDING_MODEL
    assert manifest['namespace'] == regulatory.PINECONE_NAMESPACE

    chunk_ids = [chunk['chunk_id'] for chunk in manifest['chunks']]
    assert 'ai-act-art53-1a' in chunk_ids
    assert 'cop-transparency-m1-1' in chunk_ids
    assert 'cop-transparency-m1-2' in chunk_ids
    transparency_chunks = [chunk for chunk in manifest['chunks'] if chunk['chunk_id'].startswith('cop-transparency-')]
    assert len(transparency_chunks) == 2
    assert all(chunk['text'] for chunk in transparency_chunks)
    assert all(chunk['extraction_source'] for chunk in transparency_chunks)
    assert all(chunk['status'] == 'verified' for chunk in transparency_chunks)


def test_index_regulatory_corpus_uses_stable_ids_and_namespace(monkeypatch):
    captured: dict[str, object] = {}

    def fake_describe_index(index_name, api_key, api_version=None, timeout_seconds=15.0):
        assert index_name == 'regulatory-index'
        return {'dimension': regulatory.OPENAI_EMBEDDING_DIMENSION, 'host': 'regulatory-index.svc.pinecone.io'}

    def fake_embed_texts(texts, api_key, timeout_seconds):
        captured['texts'] = list(texts)
        return [[0.1] * regulatory.OPENAI_EMBEDDING_DIMENSION for _ in texts]

    def fake_post(url, headers=None, json=None, timeout=None):
        captured['url'] = url
        captured['json'] = json
        return FakeResponse({'upsertedCount': len(json['vectors'])})

    monkeypatch.setattr(regulatory, 'describe_index', fake_describe_index)
    monkeypatch.setattr(regulatory, 'embed_texts', fake_embed_texts)
    monkeypatch.setattr(regulatory.requests, 'post', fake_post)

    report = regulatory.index_regulatory_corpus(
        {
            'OPENAI_API_KEY': 'oa-test',
            'PINECONE_API_KEY': 'pc-test',
            'PINECONE_INDEX_NAME': 'regulatory-index',
            'PINECONE_NAMESPACE': regulatory.PINECONE_NAMESPACE,
        }
    )

    verified_chunks = [chunk for chunk in regulatory.build_regulatory_corpus() if chunk.text]
    assert report['indexed'] == len(verified_chunks)
    assert report['skipped_blocked'] == 0
    assert report['compatible'] is True
    assert report['needed'] == []
    assert captured['json']['namespace'] == regulatory.PINECONE_NAMESPACE
    assert captured['json']['vectors'][0]['id'] == 'ai-act-art53-1a'
    assert captured['texts'] == [chunk.text for chunk in verified_chunks]


def test_retrieve_regulatory_context_returns_expected_provisions(monkeypatch):
    last_query: dict[str, str] = {}

    def fake_describe_index(index_name, api_key, api_version=None, timeout_seconds=15.0):
        return {'dimension': regulatory.OPENAI_EMBEDDING_DIMENSION, 'host': 'regulatory-index.svc.pinecone.io'}

    def fake_embed_texts(texts, api_key, timeout_seconds):
        last_query['text'] = texts[0]
        return [[0.2] * regulatory.OPENAI_EMBEDDING_DIMENSION]

    def fake_post(url, headers=None, json=None, timeout=None):
        if 'training' in last_query['text']:
            provision = 'Article 53(1)(d)'
            chunk_id = 'ai-act-art53-1d'
        else:
            provision = 'Article 53(1)(b)'
            chunk_id = 'ai-act-art53-1b'
        return FakeResponse(
            {
                'matches': [
                    {
                        'id': chunk_id,
                        'score': 0.91,
                        'metadata': {
                            'chunk_id': chunk_id,
                            'document_version': 'Regulation (EU) 2024/1689, OJ L, 12.7.2024',
                            'official_url': 'https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=OJ%3AL_202401689',
                            'provision': provision,
                            'excerpt': 'Excerpt',
                        },
                    }
                ]
            }
        )

    monkeypatch.setattr(regulatory, 'describe_index', fake_describe_index)
    monkeypatch.setattr(regulatory, 'embed_texts', fake_embed_texts)
    monkeypatch.setattr(regulatory.requests, 'post', fake_post)

    config = {
        'OPENAI_API_KEY': 'oa-test',
        'PINECONE_API_KEY': 'pc-test',
        'PINECONE_INDEX_NAME': 'regulatory-index',
        'PINECONE_NAMESPACE': regulatory.PINECONE_NAMESPACE,
    }

    downstream_result = regulatory.retrieve_regulatory_context(
        'official integration instructions and API call',
        config,
        item_id='US-03',
    )
    training_result = regulatory.retrieve_regulatory_context(
        'public training-content summary',
        config,
        item_id='DT-03',
    )

    assert downstream_result.hits[0].provision == 'Article 53(1)(b)'
    assert downstream_result.hits[0].chunk_id == 'ai-act-art53-1b'
    assert training_result.hits[0].provision == 'Article 53(1)(d)'
    assert training_result.hits[0].chunk_id == 'ai-act-art53-1d'
