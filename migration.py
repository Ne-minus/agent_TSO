# migrate_faiss_store.py
import os, json, pickle, types, pathlib
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document

OLD_DIR = pathlib.Path("data/scenario/vector_store")  # <-- ваша старая папка
NEW_DIR = pathlib.Path(
    "data/scenario_new/vector_store"
)  # <-- куда сохранить «чистую» версию
NEW_DIR.mkdir(parents=True, exist_ok=True)


# 1) Безопасный unpickler: подменяем любые "левые" модули на пустые классы
class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # всё, что не удаётся импортировать (например, agent.retriever.*), подменяем заглушкой
        try:
            return super().find_class(module, name)
        except Exception:
            return type(name, (), {})  # пустой класс


def safe_load_pickle(path: pathlib.Path):
    with open(path, "rb") as f:
        return SafeUnpickler(f).load()


# 2) Найдём pkl c docstore и mapping (в разных версиях название отличается)
pkl_candidates = ["index.pkl", "faiss_store.pkl", "docstore.pkl"]
pkl_path = next(
    (OLD_DIR / name for name in pkl_candidates if (OLD_DIR / name).exists()), None
)
if not pkl_path:
    raise FileNotFoundError(
        f"Не найден pickle в {OLD_DIR} ({', '.join(pkl_candidates)})"
    )

# 3) Распакуем структуру
loaded = safe_load_pickle(pkl_path)
print(loaded)

# В разных версиях это либо tuple, либо dict
if isinstance(loaded, tuple) and len(loaded) >= 2:
    index_to_docstore_id, raw_docstore = loaded[0], loaded[1]
    normalize_L2 = loaded[2] if len(loaded) >= 3 else False
elif isinstance(loaded, dict):
    index_to_docstore_id = loaded.get("index_to_docstore_id") or loaded.get(
        "index2docstore"
    )
    raw_docstore = loaded.get("docstore")
    normalize_L2 = loaded.get("normalize_L2", False)
else:
    raise RuntimeError(f"Неизвестный формат pickle: {type(loaded)}")


# 4) Превратим docstore в обычный InMemoryDocstore с Document
def to_document(obj):
    # часто это уже Document; если заглушка — пробуем извлечь поля
    if isinstance(obj, Document):
        return obj
    # предполагаемые атрибуты
    page_content = (
        getattr(obj, "page_content", None) or getattr(obj, "text", None) or ""
    )
    metadata = getattr(obj, "metadata", None) or {}
    # если это dict
    if isinstance(obj, dict):
        page_content = obj.get("page_content") or obj.get("text") or ""
        metadata = obj.get("metadata") or {}
    return Document(page_content=str(page_content), metadata=dict(metadata))


# raw_docstore может быть InMemoryDocstore или похожая структура с ._dict
if hasattr(raw_docstore, "_dict"):
    docs_map = raw_docstore._dict  # {id: Document}
elif isinstance(raw_docstore, dict):
    docs_map = raw_docstore
else:
    # как fallback попробуем атрибут data / store
    docs_map = getattr(raw_docstore, "data", None) or getattr(
        raw_docstore, "store", None
    )
    if docs_map is None:
        raise RuntimeError("Не смог извлечь словарь документов из docstore")

clean_docs = {k: to_document(v) for k, v in docs_map.items()}
docstore = InMemoryDocstore(clean_docs)

# 5) FAISS-индекс читаем напрямую (без pickle)
faiss_index_path = next(
    (OLD_DIR / n for n in ["index.faiss", "faiss.index"] if (OLD_DIR / n).exists()),
    None,
)
if not faiss_index_path:
    raise FileNotFoundError("Не найден файл FAISS индекса (index.faiss)")
index = faiss.read_index(str(faiss_index_path))

# 6) Собираем новый VectorStore с вашим текущим embeddings
#    ВАЖНО: импортируйте ваш get_embeddings здесь
from agent.model.model_init import get_embeddings

emb = get_embeddings()

vs = FAISS(
    embedding_function=emb,
    index=index,
    docstore=docstore,
    index_to_docstore_id=index_to_docstore_id,
)

# 7) Сохраняем «чистую» копию
vs.save_local(str(NEW_DIR))
print("Готово: сохранён чистый FAISS store в", NEW_DIR)
