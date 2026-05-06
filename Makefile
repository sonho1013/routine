.PHONY: install verify run reset wheels warmup test clean

install:
	bash scripts/setup.sh

verify:
	bash scripts/verify.sh

run:
	bash scripts/run_demo.sh

reset:
	bash scripts/reset_demo.sh

wheels:
	bash scripts/download_wheels.sh

warmup:
	LLM_CACHE_READ_ONLY= python scripts/warmup_llm_cache.py

test:
	pytest tests/ -x -q --ignore=tests/test_kling_omnivideo.py --ignore=tests/test_storyboard_generation.py --ignore=tests/test_embedding_distance.py --ignore=tests/test_eps_sweep.py

clean:
	rm -rf .venv storage/habit_memory.db* storage/memories storage/llm_cache.json storage/embedding_cache.json
