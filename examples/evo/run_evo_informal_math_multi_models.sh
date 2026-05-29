set -euo pipefail

export no_proxy=localhost,127.0.0.1
export NO_PROXY=localhost,127.0.0.1
export HF_ENDPOINT=https://hf-mirror.com

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/alphaapollo/core/generation:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

data_source="math-ai/aime24"

python3 -m alphaapollo.data_preprocess.prepare_evolving_data --data_source $data_source --local_dir ./data

python3 -m alphaapollo.core.generation.evolving.evolving_multi_models --config examples/configs/vllm_informal_math_multi_models.yaml
