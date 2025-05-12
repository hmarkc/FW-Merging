#!/bin/bash

# Create the which12 directory if it doesn't exist
mkdir -p which12

# Download each model using huggingface-cli
huggingface-cli download  --local-dir which12/Synthia-7B-v1.2 migtissera/Synthia-7B-v1.2
echo "Downloaded Synthia-7B-v1.2"

huggingface-cli download  --local-dir which12/Llama-2-7b-evolcodealpaca neuralmagic/Llama-2-7b-evolcodealpaca
echo "Downloaded Llama-2-7b-evolcodealpaca"

huggingface-cli download  --local-dir which12/OpenHermes-7B teknium/OpenHermes-7B
echo "Downloaded OpenHermes-7B"

huggingface-cli download  --local-dir which12/pygmalion-2-7b PygmalionAI/pygmalion-2-7b
echo "Downloaded pygmalion-2-7b"

huggingface-cli download  --local-dir which12/Llama-2-7b-chat-hf meta-llama/Llama-2-7b-chat-hf
echo "Downloaded Llama-2-7b-chat-hf"

huggingface-cli download  --local-dir which12/BeingWell_llama2_7b Severus27/BeingWell_llama2_7b
echo "Downloaded BeingWell_llama2_7b"

huggingface-cli download  --local-dir which12/MetaMath-7B-V1.0 meta-math/MetaMath-7B-V1.0
echo "Downloaded MetaMath-7B-V1.0"

huggingface-cli download  --local-dir which12/vicuna-7b-v1.5 lmsys/vicuna-7b-v1.5
echo "Downloaded vicuna-7b-v1.5"

huggingface-cli download  --local-dir which12/Platypus2-7B garage-bAInd/Platypus2-7B
echo "Downloaded Platypus2-7B"

huggingface-cli download  --local-dir which12/GOAT-7B-Community GOAT-AI/GOAT-7B-Community
echo "Downloaded GOAT-7B-Community"

huggingface-cli download  --local-dir which12/Llama-2-7b-WikiChat-fused stanford-oval/Llama-2-7b-WikiChat-fused
echo "Downloaded Llama-2-7b-WikiChat-fused"

huggingface-cli download  --local-dir which12/dolphin-llama2-7b cognitivecomputations/dolphin-llama2-7b
echo "Downloaded dolphin-llama2-7b" 