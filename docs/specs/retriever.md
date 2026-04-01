## Retriever Spec (Attractions)

- Источники: Спарсенные списки топ-достопримечательностей из Wikipedia/WikiVoyage (для PoC).
- Индекс: VectorDB (Chroma/Qdrant). Chunking size: 512 токенов.
- Поиск: Cosine similarity, top-K = 5.
- Ограничения: Векторный поиск работает только если город известен агенту. Если город не найден в БД, возвращается пустой список (graceful degradation), и агент использует внутренние знания LLM.
