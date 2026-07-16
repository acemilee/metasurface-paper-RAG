import threading
import time

from paper_rag.embedding_server import PriorityInferenceScheduler


def test_query_runs_before_waiting_ingestion_batch() -> None:
    scheduler = PriorityInferenceScheduler()
    first_started = threading.Event()
    release_first = threading.Event()
    order = []

    def first_ingestion() -> None:
        def operation():
            first_started.set()
            release_first.wait(timeout=2)
            order.append("ingestion-1")
        scheduler.run("ingestion", operation)

    def second_ingestion() -> None:
        scheduler.run("ingestion", lambda: order.append("ingestion-2"))

    def query() -> None:
        scheduler.run("query", lambda: order.append("query"))

    threads = [
        threading.Thread(target=first_ingestion),
        threading.Thread(target=second_ingestion),
        threading.Thread(target=query),
    ]
    threads[0].start()
    assert first_started.wait(timeout=1)
    threads[1].start()
    time.sleep(0.05)
    threads[2].start()
    time.sleep(0.05)
    release_first.set()
    for thread in threads:
        thread.join(timeout=2)

    assert order == ["ingestion-1", "query", "ingestion-2"]
