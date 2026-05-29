set -x
ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=XFORMERS

export no_proxy=localhost,127.0.0.1
export NO_PROXY=localhost,127.0.0.1

data_source='DigitalLearningGmbH/MATH-lighteval'
# bash arrays are space-separated, not comma-separated; one item per line.
test_data_sources=(
    'HuggingFaceH4/MATH-500'
    'math-ai/aime24'
    'math-ai/aime25'
    'math-ai/amc23'
    'math-ai/olympiadbench'
    'math-ai/minervamath'
    'openai/gsm8k:main'
)
# one repeat count per dataset (must match len(test_data_sources)).
test_repeats=(32 32 32 32 32 32 32)

# Prepare data
python3 -m alphaapollo.data_preprocess.prepare_rl_training_data \
    --data_source $data_source

python3 -m alphaapollo.data_preprocess.prepare_merged_validation_data \
    --data_sources "${test_data_sources[@]}" \
    --splits test \
    --repeat "${test_repeats[@]}"
