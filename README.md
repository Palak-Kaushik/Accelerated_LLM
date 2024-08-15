# Accelerated-LLM

Accelerated LLM utilizes multiple small LLMs instead of a single large LLM, each specialised in a singular task to reduce resources required while preserving the response quality.


- Single LLMs struggle to handle complex, multi-domain queries effectively. 
- General-purpose models lack specialized knowledge needed for domains like e-commerce, travel, fintech, mobile, and media.
- Single LLMs often consume significant resources without delivering optimized results for specific tasks.

## Proposed Architecture

- **Query Router LLM**: Analyzes user queries and reroutes them to already trained domain-specific Sub-LLMs.
- **Sub-LLMs**: Tailored to handle specific types of queries to provide more accurate and relevant responses. Trained via knowledge distillation to reduce hallucinations
- **Agentic RAG**: Each Sub-LLM contains multiple specialized agents using Retrieval Augmented Generation to handle specific tasks.




<img src="https://github.com/user-attachments/assets/390171f4-d76d-4ec5-9930-36eeac50e1b4" alt="architecture of accelerated llm" style="width:75%;"/>


## Features of Accelerated LLM


1. **Reducing Memory Overhead**: Using smaller language models that are fine-tuned for a specific task are faster at performing given task and use considerably less memory (improving efficiency upto 50%)[^1] , making this approach much more scalable than existing solutions.

1. **Reducing Hallucinations**: Hallucinations can be reduced by using knowledge distillation[^2], as well as by using multiple agents within a LLM. This solution will generate more accurate responses and will not return wrong data.

1. **Preserving quality responses**: Using multiple lightweight language models fine-tuned for particular tasks give quality responses as compared to using a single large model. Using multiple agents allows the LLM to generate detailed responses and ensures that wrong information is not returned in the response. Multiple agents also allow for parallel processing, enabling multiple logical tasks to be performed simultaneously.
 
1. **Can be used for wide variety of purposes**: The sub-LLMs can we be fine-tuned and used for a wide variety of tasks, including but not limited to fintech, ecommerce, mobile applications and many more.
 
1. **Data Security**: Use of open source LLMs by the companies will ensure that their data is private and not being shared with a third party.
 
1. **Faster Response**: Smaller LLMs can generate responses much faster as compared to their larger counterparts.
 
1. **Feasibility**: Reduces costs due to the reduced model size and time required for response generation.
 
1. **Scalability**: Increased scalability due to reduced memory overhead.


[^1]: Ong, Isaac, et al. "RouteLLM: Learning to Route LLMs with Preference Data." arXiv preprint arXiv:2406.18665 (2024).
[^2]: Daniel McDonald, Rachael Papadopoulos, Leslie Benningfield. Reducing LLM Hallucination Using Knowledge Distillation: A Case Study with Mistral Large and MMLU Benchmark. TechRxiv. May 25, 2024. DOI: 10.36227/techrxiv.171665607.76504195/v1

